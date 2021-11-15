# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import json

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator
from utils import is_spatial_question

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


NUM_OBJ_PERMUTATIONS = [2, 4, 9, 18, 36]


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # Transfer model to GPU before apex.
        self.model = self.model.cuda()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)


        # Half Precision 
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optim = amp.initialize(self.model, self.optim, opt_level='O2')
        
        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                if args.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), 5.)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)

                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}

        differences = {p: 0 for p in NUM_OBJ_PERMUTATIONS}
        spatial_differences = {p: 0 for p in NUM_OBJ_PERMUTATIONS}
        num_questions = 0
        num_spatial_questions = 0

        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()

                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

                # Run inference with permuted bounding boxes, if necessary.
                if args.permute_bbox:

                    # Determine which questions involve spatial descriptors.
                    spatial_q = torch.zeros_like(label)
                    for j, question in enumerate(sent):
                        if is_spatial_question(question):
                            spatial_q[j] = 1

                    num_questions += feats.shape[0]
                    num_spatial_questions += spatial_q.sum().item()

                    for num_permuted in NUM_OBJ_PERMUTATIONS:

                        # Permute bounding boxes, making sure that we do not send any
                        # object back to its original position.
                        num_objs = boxes.shape[1]
                        objs = torch.randperm(num_objs, device=boxes.device)
                        objs = objs[:num_permuted]
                        perm = torch.randperm(num_permuted, device=boxes.device)
                        eye = torch.arange(num_permuted, device=boxes.device)
                        while torch.any(perm == eye):
                            perm = torch.randperm(num_permuted, device=boxes.device)
                        permuted = objs[perm]
                        total_perm = torch.arange(num_objs, device=boxes.device)
                        total_perm[objs] = permuted
                        permuted_boxes = boxes[:, total_perm]

                        # Pass permuted inputs to model and count differences against
                        # the original predictions.
                        p_logit = self.model(feats, permuted_boxes, sent)
                        p_score, p_label = p_logit.max(1)
                        diff = (p_label != label).sum().item()
                        s_diff = torch.logical_and(p_label != label, spatial_q)
                        s_diff = s_diff.sum().item()
                        differences[num_permuted] += diff
                        spatial_differences[num_permuted] += s_diff

        # Store differences in model prediction from permuting a varying number of
        # bounding boxes.
        if args.permute_bbox:
            differences = {k: v / num_questions for k, v in differences.items()}
            spatial_differences = {
                k: v / num_spatial_questions for k, v in spatial_differences.items()
            }
            total_diffs = {
                "differences": differences,
                "spatial_differences": spatial_differences,
            }
            with open("permutation_probe.json", "w") as probe_file:
                json.dump(total_diffs, probe_file, indent=4)

        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        score, sr_score = eval_tuple.evaluator.evaluate(
            quesid2ans, spatial_reasoning=True
        )
        return score, sr_score

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            accuracy, sr_accuracy = vqa.evaluate(
                get_data_tuple('minival', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(f"Total accuracy: {accuracy}")
            print(f"SR accuracy: {sr_accuracy}")
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)


