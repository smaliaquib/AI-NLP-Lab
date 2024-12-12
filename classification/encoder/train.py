import os
import torch 
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from model_dispatcher import MODEL_DISPATCHER
from torcheval import metrics
from dataset import TextClassification

load_dotenv()

ES = int(os.environ.get("EARLY_STOP"))
CROSS_VAL = os.environ.get("CROSS_VAL")
MAX_LEN = int(os.environ.get("MAX_LEN"))
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
TRAINING_CSV = os.environ.get("TRAINING_CSV")
EPOCHS = int(os.environ.get("EPOCHS"))
BASE_MODEL = os.environ.get("BASE_MODEL")
DEVICE = os.environ.get("DEVICE")
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE"))

# TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
# VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get("BASE_MODEL")
selected_config = MODEL_DISPATCHER[BASE_MODEL]

print(f"Model will run on {DEVICE}")

def score_accuracy(predicted, y):
    try:
        return accuracy_score(predicted, y)
    except ValueError:
        return 0.0

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def train(training_loader, model, optimizer, loss_function, metric, cost_metric):
    cost = 0
    metric.reset()
    cost_metric.reset()
    model.train()

    loop = tqdm(enumerate(training_loader), total=len(training_loader), desc="\033[94mTraining\033[0m", ncols=100,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] |{postfix}", colour="blue")
    
    for batch_idx, data in loop:
        ids = data['ids'].to(DEVICE, dtype = torch.long)
        mask = data['mask'].to(DEVICE, dtype = torch.long)
        targets = data['targets'].to(DEVICE, dtype = torch.long)
        optimizer.zero_grad()

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        
        metric.update(outputs, targets)
        cost_metric.update(loss.detach(), weight=len(ids))
        acc = metric.compute()
        cost = cost_metric.compute()
        
        loop.set_postfix(loss= f"\033[94m{cost:.4f}\033[0m", f1= f"\033[94m{acc:.4f}\033[0m", 
        lr=f"\033[94m{optimizer.param_groups[0]['lr']:.6f}\033[0m") 
    return cost, acc



def evaluate(testing_loader, model, optimizer, loss_function, metric, cost_metric):

    cost = 0
    metric.reset()
    cost_metric.reset()
    model.eval()
    with torch.no_grad():
        loop = tqdm(enumerate(testing_loader), total=len(testing_loader), desc="\033[93mValidation\033[0m", ncols=100,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] |{postfix}", colour="yellow")
        
        for batch_idx, data in loop:
            ids = data['ids'].to(DEVICE, dtype = torch.long)
            mask = data['mask'].to(DEVICE, dtype = torch.long)
            targets = data['targets'].to(DEVICE, dtype = torch.long)
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)

            metric.update(outputs, targets)
            cost_metric.update(loss.detach(), weight=len(ids))
            acc = metric.compute()
            cost = cost_metric.compute()

            loop.set_postfix(loss= f"\033[93m{cost:.4f}\033[0m", f1= f"\033[93m{acc:.4f}\033[0m", 
            lr=f"\033[93m{optimizer.param_groups[0]['lr']:.6f}\033[0m")

    return cost, acc

def main(kfold, es):

    tokenizer = selected_config['tokenizer'].from_pretrained(selected_config['pretrained_name'], truncation=True, model_max_length= MAX_LEN)

    if kfold:
        data = pd.read_csv(TRAINING_FOLDS_CSV)
        fold = 1

        df_train = data[data.kfold != fold].reset_index(drop=True)
        df_valid = data[data.kfold == fold].reset_index(drop=True)
    else:
        data = pd.read_csv(TRAINING_CSV)
        train_size = 0.8
        df_train= data.sample(frac=train_size, random_state=200)
        df_valid= data.drop(df_train.index).reset_index(drop=True)
        df_train = df_train.reset_index(drop=True)

    NUM_CLASSES = data.rating.nunique()
    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(df_train.shape))
    print("TEST Dataset: {}".format(df_valid.shape))
    training_set = TextClassification(df_train, tokenizer, MAX_LEN)
    testing_set = TextClassification(df_valid, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    valid_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }


    train_loader = torch.utils.data.DataLoader(training_set, **train_params)
    valid_loader = torch.utils.data.DataLoader(testing_set, **valid_params)

    model = selected_config['model']()
    model.to(DEVICE)
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    metric = metrics.MulticlassPrecisionRecallCurve(num_classes=NUM_CLASSES)
    metric = metrics.MulticlassAccuracy(num_classes=NUM_CLASSES, device=DEVICE)
    metric = metrics.MulticlassAUROC(num_classes=NUM_CLASSES, device=DEVICE)
    metric = metrics.MulticlassF1Score(num_classes=NUM_CLASSES, average="micro", device=DEVICE)
    cost_metric = metrics.Mean(device=DEVICE)
    tr_cost = []
    ts_cost = []
    tr_acc = []
    ts_acc = []
    early_stop_thresh = es
    best_score = -1
    best_epoch = -1
    for epoch in range(EPOCHS):
        cost_train, tracc = train(train_loader, model, optimizer, loss_function, metric, cost_metric)
        cost_val, tsacc = evaluate(valid_loader, model, optimizer, loss_function, metric, cost_metric) 

        tr_cost.append(cost_train)
        ts_cost.append(cost_val) 
        tr_acc.append(tracc)
        ts_acc.append(tsacc)
        if tsacc > best_score:
            best_score = tsacc
            best_epoch = epoch
            checkpoint(model, f"save_dict/best_model_{fold}.pth")
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break  # terminate the training loop
    return tr_cost, ts_cost, tr_acc, ts_acc, best_score

if __name__ == "__main__":
    main(kfold=CROSS_VAL, es = ES)
