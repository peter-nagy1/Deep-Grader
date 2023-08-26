# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning Deep Grader and Deep Siamese Grader models on the task of
automatic program grading  with Python and C++ programming languages.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments, Trainer
from datasets import load_metric
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from tqdm import tqdm, trange

# Ignores disruptive warnings which occur when using multiple gpus
import warnings
warnings.filterwarnings(action='ignore')

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="The model to train. (deep_grader, deep_siamese_grader)")
    parser.add_argument("--encoder_name", default=None, type=str, required=True,
                        help="The encoder checkpoint used in the model's architecture.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--language", default=None, type=str, required=True,
                        help="The programming language used. (python, cpp)")
    parser.add_argument("--setting", default=None, type=str, required=True,
                        help="The question splitting setting used. (independent, dependent)")

    # Other parameters  
    parser.add_argument('--cache_dir', default=None, type=str,
                        help="The cache directory for Huggingface models.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--margin", default=2.0, type=float,
                        help="The margin used in the contrastive loss.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', default=12, type=int,
                        help="The random seed for initialization.")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    #set_seed(args.seed)

    # Get data
    train = pd.read_pickle(f"data/{args.language}/{args.setting}/train.pkl", compression="zip")
    validate = pd.read_pickle(f"data/{args.language}/{args.setting}/validate.pkl", compression="zip")
    test = pd.read_pickle(f"data/{args.language}/{args.setting}/test.pkl", compression="zip")
    graded = pd.read_pickle(f"data/{args.language}/graded.pkl", compression="zip")
    unpaired_test = pd.read_pickle(f"data/{args.language}/unpaired_test.pkl", compression="zip")
    unpaired_graded = pd.read_pickle(f"data/{args.language}/unpaired_graded.pkl", compression="zip")

    # Retrieve the tokenizer from Huggingface
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, cache_dir=args.cache_dir)


    # Deep Grader Model
    if args.model_name == "deep_grader":

        from Deep_Grader import Dataset, tokenize, getMajorityVote, getMajorityVoteGrade

        # Retrieve the encoder from Huggingface
        encoder = AutoModelForSequenceClassification.from_pretrained(args.encoder_name, num_labels=2, cache_dir=args.cache_dir)

        # Tokenize data
        tk_train = tokenize(train['sub'].tolist(), tokenizer)
        tk_validate = tokenize(validate['sub'].tolist(), tokenizer)
        tk_test = tokenize(test['sub'].tolist(), tokenizer)

        # Create datasets
        train_dataset = Dataset(tk_train, train['label'].tolist())
        val_dataset = Dataset(tk_validate, validate['label'].tolist())
        test_dataset = Dataset(tk_test, test['label'].tolist())

        # Set training parameters
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            load_best_model_at_end= True,
        )

        # Create trainer
        trainer = Trainer(
            model=encoder,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train model
        trainer.train()

        # Save model
        trainer.save_model(f"{args.output_dir}/{args.model_name}")

        # Evaluate model
        print("Weak label evaluation:")
        pred = trainer.predict(test_dataset)
        preds, labels = np.argmax(pred.predictions, axis=1), pred.label_ids
        acc, f1 = [f(labels, preds) for f in [accuracy_score, f1_score]]
        print(f"Accuracy: %.3f - F1: %.3f" % (acc, f1))

        # Evaluating graded data problems individually
        print("Soft label evaluation:")
        gradedPID = np.unique(graded['pId'])
        results = []
        for prob in gradedPID:
            problem = graded[graded['pId'] == prob]
            tk_labeled = tokenize(problem['sub'].tolist(), tokenizer)
            labeled_dataset = Dataset(tk_labeled, problem['label'].tolist())
            
            preds = trainer.predict(labeled_dataset).predictions[:,1]
            normalized_preds = (preds-min(preds))/(max(preds)-min(preds)) * (5-1) + 1  # Put between 0-1 and scale up to 1-5
            labels = problem['grade'].tolist()
            corr, mse = [f(labels, normalized_preds) for f in [np.corrcoef, mean_squared_error]]
            
            results.append([corr[0][1], mse])
        corr, mse = np.mean(results, axis=0)
        print(f"\nCorrelation: %.3f - MSE: %.3f" % (corr, mse))

        # Majority Vote
        print("Majority Vote weak label evaluation:")
        testPID = np.unique(unpaired_test['pId'])
        preds, labels = [], []
        for prob in tqdm(testPID):
            correctSet = unpaired_test[(unpaired_test['pId'] == prob) & (unpaired_test['label'] == 1)]
            indices = unpaired_test[unpaired_test['pId'] == prob].index
            
            for index in indices:
                pred = getMajorityVote(unpaired_test.at[index, 'sub'], correctSet['sub'].drop(index, errors='ignore'), tokenizer, trainer)
                preds.append(pred)
                labels.append(unpaired_test.at[index, 'label'])
        acc, f1 = [f(labels, preds) for f in [accuracy_score, f1_score]]
        print(f"Accuracy: %.3f - F1: %.3f" % (acc, f1))

        # Graded Majority Vote
        print("Majority Vote soft label evaluation:")
        results = []
        for prob in tqdm(gradedPID):
            problem = unpaired_graded[unpaired_graded['pId'] == prob]
            correctSet = problem[problem['label'] == 1]
            indices = problem.index
            
            preds = []
            for index in indices:
                preds.append( getMajorityVoteGrade(problem.at[index, 'sub'], correctSet['sub'].drop(index, errors='ignore'), tokenizer, trainer) )
                
            normalized_preds = (preds-min(preds))/(max(preds)-min(preds)) * (5-1) + 1  # Put between 0-1 and scale up to 1-5
            labels = problem['grade'].tolist()
            corr, mse = [f(labels, normalized_preds) for f in [np.corrcoef, mean_squared_error]]
            
            results.append([corr[0][1], mse])
        corr, mse = np.mean(results, axis=0)
        print(f"Correlation: %.3f - MSE: %.3f" % (corr, mse))

        # Incremental
        print("Incremental weak label evaluation:")
        unpaired_test = unpaired_test.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        preds, labels = [], []
        for prob in tqdm(testPID):
            correctSet = unpaired_test[(unpaired_test['pId'] == prob) & (unpaired_test['label'] == 1)][:1]  # Start with the first correct sub
            indices = unpaired_test[unpaired_test['pId'] == prob].index
            correctSet_prevlen = 0

            while len(correctSet) != correctSet_prevlen:
                iter_preds, iter_labels = [], []
                correctSet_prevlen = len(correctSet)

                for index in indices:
                    cSet = correctSet['sub'].drop(index, errors='ignore')
                    if len(cSet) == 0:
                        pred = 1
                    else:
                        pred = getMajorityVote(unpaired_test.at[index, 'sub'], cSet, tokenizer, trainer)

                    iter_preds.append(pred)
                    iter_labels.append(unpaired_test.at[index, 'label'])

                    if pred == 1 and index not in correctSet.index:
                        correctSet = correctSet.append(unpaired_test.loc[index])
            
            preds.extend(iter_preds)
            labels.extend(iter_labels)
        acc, f1 = [f(labels, preds) for f in [accuracy_score, f1_score]]
        print(f"Accuracy: %.3f - F1: %.3f" % (acc, f1))

        # Graded Incremental
        print("Incremental soft label evaluation:")
        unpaired_graded = unpaired_graded.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        results = []
        for prob in tqdm(gradedPID):
            problem = unpaired_graded[unpaired_graded['pId'] == prob]
            correctSet = problem[problem['label'] == 1][:1]  # Start with the first correct sub
            indices = problem.index
            correctSet_prevlen = 0

            while len(correctSet) != correctSet_prevlen:
                iter_preds, iter_labels = [], []
                correctSet_prevlen = len(correctSet)

                for index in indices:
                    predL = None
                    cSet = correctSet['sub'].drop(index, errors='ignore')
                    if len(cSet) == 0:
                        predL = 1
                        pred = 1
                    else:
                        predL = getMajorityVote(problem.at[index, 'sub'], cSet, tokenizer, trainer)
                        pred = getMajorityVoteGrade(problem.at[index, 'sub'], cSet, tokenizer, trainer)

                    iter_preds.append(pred)
                    iter_labels.append(problem.at[index, 'grade'])
                    
                    if predL == 1 and index not in correctSet.index:
                        correctSet = correctSet.append(problem.loc[index]) 
            
            normalized_preds = (iter_preds-min(iter_preds))/(max(iter_preds)-min(iter_preds)) * (5-1) + 1  # Put between 0-1 and scale up to 1-5
            labels = iter_labels
            corr, mse = [f(labels, normalized_preds) for f in [np.corrcoef, mean_squared_error]]
            
            results.append([corr[0][1], mse])
        corr, mse = np.mean(results, axis=0)
        print(f"Correlation: %.3f - MSE: %.3f" % (corr, mse))


    # Deep Siamese Grader Model
    elif args.model_name == "deep_siamese_grader":

        from Deep_Siamese_Grader import Dataset, tokenize, SiameseNetwork, ContrastiveLoss, testing, getMajorityVote, getMajorityVoteGrade

        # Retrieve the encoder from Huggingface
        encoder = AutoModel.from_pretrained(args.encoder_name, cache_dir=args.cache_dir)

        # Tokenize data
        tk_train = tokenize(train['sub'].tolist(), tokenizer)
        tk_validate = tokenize(validate['sub'].tolist(), tokenizer)
        tk_test = tokenize(test['sub'].tolist(), tokenizer)

        # Create datasets
        train_dataset = Dataset(tk_train[0], tk_train[1], train['label'].tolist())
        val_dataset = Dataset(tk_validate[0], tk_validate[1], validate['label'].tolist())
        test_dataset = Dataset(tk_test[0], tk_test[1], test['label'].tolist())

        # Train model
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4*args.train_batch_size)
        model = torch.nn.DataParallel(SiameseNetwork(encoder).cuda())
        criterion = ContrastiveLoss(margin=args.margin)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        for epoch in trange(0, args.num_train_epochs):
            for i, data in enumerate(tqdm(train_dataloader, leave=False), 0):
                sub1, attn1 = data[0].values()
                sub2, attn2 = data[1].values()
                label = data[2]
                sub1, attn1 = Variable(sub1).cuda(), Variable(attn1).cuda()
                sub2, attn2 = Variable(sub2).cuda(), Variable(attn2).cuda()
                label = Variable(label).cuda()
                output1, output2 = model((sub1, attn1), (sub2, attn2))
                optimizer.zero_grad()
                loss_contrastive = criterion(output1, output2, label)
                loss_contrastive.backward()
                optimizer.step()

        # Save model
        torch.save(model.state_dict(), f"{args.output_dir}/{args.model_name}")

        # Evaluate model
        print("Weak label evaluation:")
        dataloader = DataLoader(test_dataset, batch_size=2*args.eval_batch_size)
        preds, labels = testing(dataloader, model)
        bin_preds = 1 / np.exp(preds) < 0.5
        acc, f1 = [f(labels, bin_preds) for f in [accuracy_score, f1_score]]
        print(f"Accuracy: %.3f - F1: %.3f" % (acc, f1))

        # Evaluating graded data problems individually
        print("Soft label evaluation:")
        gradedPID = np.unique(graded['pId'])
        results = []
        for prob in gradedPID:
            problem = graded[graded['pId'] == prob]
            tk_graded = tokenize(problem['sub'].tolist(), tokenizer)
            graded_dataset = Dataset(tk_graded[0], tk_graded[1], problem['grade'].tolist())
            dataloader = DataLoader(graded_dataset, batch_size=2*args.eval_batch_size)
            
            preds, grades = testing(dataloader, model)
            preds = np.array(preds)
            preds = (1-(preds-min(preds))/(max(preds)-min(preds))) * (5-1) + 1
            corr, mse = [f(grades, preds) for f in [np.corrcoef, mean_squared_error]]
            
            results.append([corr[0][1], mse])
        corr, mse = np.mean(results, axis=0)
        print(f"Correlation: %.3f - MSE: %.3f" % (corr, mse))

        # Majority Vote
        print("Majority Vote weak label evaluation:")
        testPID = np.unique(unpaired_test['pId'])
        preds, labels = [], []
        for prob in tqdm(testPID):
            correctSet = unpaired_test[(unpaired_test['pId'] == prob) & (unpaired_test['label'] == 1)]
            indices = unpaired_test[unpaired_test['pId'] == prob].index
            
            for index in indices:
                pred = getMajorityVote(unpaired_test.at[index, 'sub'], correctSet['sub'].drop(index, errors='ignore'), tokenizer, 4*args.eval_batch_size, model)
                preds.append(pred)
                labels.append(unpaired_test.at[index, 'label'])
        acc, f1 = [f(labels, preds) for f in [accuracy_score, f1_score]]
        print(f"Accuracy: %.3f - F1: %.3f" % (acc, f1))

        # Graded Majority Vote
        print("Majority Vote soft label evaluation:")
        results = []
        for prob in tqdm(gradedPID):
            problem = unpaired_graded[unpaired_graded['pId'] == prob]
            correctSet = problem[problem['label'] == 1]
            indices = problem.index
            
            preds = []
            for index in indices:
                preds.append( getMajorityVoteGrade(problem.at[index, 'sub'], correctSet['sub'].drop(index, errors='ignore'), tokenizer, 4*args.eval_batch_size, model) )

            preds = np.array(preds)    
            normalized_preds = (1-(preds-min(preds))/(max(preds)-min(preds))) * (5-1) + 1  # Put between 0-1 and scale up to 1-5
            labels = problem['grade'].tolist()
            corr, mse = [f(labels, normalized_preds) for f in [np.corrcoef, mean_squared_error]]
            
            results.append([corr[0][1], mse])
        corr, mse = np.mean(results, axis=0)
        print(f"Correlation: %.3f - MSE: %.3f" % (corr, mse))

        # Incremental
        print("Incremental weak label evaluation:")
        unpaired_test = unpaired_test.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        preds, labels = [], []
        for prob in tqdm(testPID):
            correctSet = unpaired_test[(unpaired_test['pId'] == prob) & (unpaired_test['label'] == 1)][:1]  # Start with the first correct sub
            indices = unpaired_test[unpaired_test['pId'] == prob].index
            correctSet_prevlen = 0

            while len(correctSet) != correctSet_prevlen:
                iter_preds, iter_labels = [], []
                correctSet_prevlen = len(correctSet)

                for index in indices:
                    cSet = correctSet['sub'].drop(index, errors='ignore')
                    if len(cSet) == 0:
                        pred = 1
                    else:
                        pred = getMajorityVote(unpaired_test.at[index, 'sub'], cSet, tokenizer, 4*args.eval_batch_size, model)

                    iter_preds.append(pred)
                    iter_labels.append(unpaired_test.at[index, 'label'])

                    if pred == 1 and index not in correctSet.index:
                        correctSet = correctSet.append(unpaired_test.loc[index])
            
            preds.extend(iter_preds)
            labels.extend(iter_labels)
        acc, f1 = [f(labels, preds) for f in [accuracy_score, f1_score]]
        print(f"Accuracy: %.3f - F1: %.3f" % (acc, f1))

        # Graded Incremental
        print("Incremental soft label evaluation:")
        unpaired_graded = unpaired_graded.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        results = []
        for prob in tqdm(gradedPID):
            problem = unpaired_graded[unpaired_graded['pId'] == prob]
            correctSet = problem[problem['label'] == 1][:1]  # Start with the first correct sub
            indices = problem.index
            correctSet_prevlen = 0

            while len(correctSet) != correctSet_prevlen:
                iter_preds, iter_labels = [], []
                correctSet_prevlen = len(correctSet)

                for index in indices:
                    predL = None
                    cSet = correctSet['sub'].drop(index, errors='ignore')
                    if len(cSet) == 0:
                        predL = 1
                        pred = 1
                    else:
                        predL = getMajorityVote(problem.at[index, 'sub'], cSet, tokenizer, 2*args.eval_batch_size, model)
                        pred = getMajorityVoteGrade(problem.at[index, 'sub'], cSet, tokenizer, 2*args.eval_batch_size, model)

                    iter_preds.append(pred)
                    iter_labels.append(problem.at[index, 'grade'])
                    
                    if predL == 1 and index not in correctSet.index:
                        correctSet = correctSet.append(problem.loc[index]) 
            
            iter_preds = np.array(iter_preds)
            normalized_preds = (1-(iter_preds-min(iter_preds))/(max(iter_preds)-min(iter_preds))) * (5-1) + 1  # Put between 0-1 and scale up to 1-5
            labels = iter_labels
            corr, mse = [f(labels, normalized_preds) for f in [np.corrcoef, mean_squared_error]]
            
            results.append([corr[0][1], mse])
        corr, mse = np.mean(results, axis=0)
        print(f"Correlation: %.3f - MSE: %.3f" % (corr, mse))

if __name__ == "__main__":
    main()
