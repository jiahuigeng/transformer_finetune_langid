import torch
import re
import pandas as pd
from datasets import Dataset
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import Trainer, TrainingArguments

import argparse



def clean_html(text):
  html = re.compile('<.*?>')#regex
  return html.sub(r'',text)

def remove_links(tweet):
  '''Takes a string and removes web links from it'''
  tweet = re.sub(r'http\S+', '', tweet) # remove http links
  tweet = re.sub(r'bit.ly/\S+', '', tweet) # remove bitly links
  tweet = tweet.strip('[link]') # remove [links]
  return tweet

def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pat, '', text)

def remove_(tweet):
  tweet = re.sub('([_]+)', "", tweet)
  return tweet

def punct(text):
  token=RegexpTokenizer(r'\w+')#regex
  text = token.tokenize(text)
  text= " ".join(text)
  return text

def email_address(text):
  email = re.compile(r'[\w\.-]+@[\w\.-]+')
  return email.sub(r'',text)

def lower(text):
  return text.lower()

def removeStopWords(str):
  cachedStopWords = set(stopwords.words("english"))
  cachedStopWords.update(('and','I','A','http','And','So','arnt','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these','mailto','regards','ayanna','like','email'))
  new_str = ' '.join([word for word in str.split() if word not in cachedStopWords])
  return new_str


def text_preprocess(data, col):
    data[col] = data[col].apply(func=clean_html)
    data[col] = data[col].apply(func=remove_)
    data[col] = data[col].apply(func=removeStopWords)
    data[col] = data[col].apply(func=remove_links)
    data[col] = data[col].apply(func=remove_special_characters)
    data[col] = data[col].apply(func=punct)
    data[col] = data[col].apply(func=email_address)
    data[col] = data[col].apply(func=lower)
    return data

df = pd.read_csv("Language Detection.csv")
df.Language.replace(to_replace=['Portugeese', 'Sweedish'], value=['Portuguese', 'Swedish'], inplace=True)
df = df[df.Language.isin(
    ["English", "French", "Dutch", "Spanish", "Danish", "Italian", "Swedish", "German", "Portuguese", "Turkish"])]

model_ckpt = "smallbenchnlp/roberta-small"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 125

preprocessed_df = text_preprocess(df,'Text')

num_labels = len(df.Language.unique())
labels_dict_reverse = {}
labels_dict = {}
for idx, lang in enumerate(preprocessed_df.Language.unique()):
    labels_dict[lang] = idx
    labels_dict_reverse[str(idx)] = lang

print(labels_dict)
print(labels_dict_reverse)
preprocessed_df['label'] = preprocessed_df.Language.map(labels_dict)

train_df, test_df = train_test_split(preprocessed_df,test_size=0.3,random_state=42,shuffle=True,stratify=preprocessed_df.Language)


def tokenize(batch):
    return tokenizer(batch["Text"], padding=True, truncation=True)

train_set = Dataset.from_pandas(train_df)
test_set = Dataset.from_pandas(test_df)
train_encoded = train_set.map(tokenize, batched=True)
test_encoded = test_set.map(tokenize, batched=True)

train_encoded.set_format("torch", ["input_ids", "attention_mask", "label"])
test_encoded.set_format("torch", ["input_ids", "attention_mask", "label"])


def main(args):


    if args.train_and_test:
        batch_size = 64
        logging_steps = len(train_df.Text.tolist()) // batch_size

        model_name = f"{model_ckpt}-finetuned-langid"
        training_args = TrainingArguments(output_dir=model_name,
                                          num_train_epochs=args.num_train_epochs,
                                          learning_rate=2e-5,
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          weight_decay=0.01,
                                          evaluation_strategy="epoch",
                                          disable_tqdm=False,
                                          logging_steps=logging_steps,
                                          push_to_hub=False,
                                          log_level="error")

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            f1 = f1_score(labels, preds, average="weighted")
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc, "f1": f1}


        model = (AutoModelForSequenceClassification
                 .from_pretrained(model_ckpt, num_labels=num_labels)
                 .to(device))


        trainer = Trainer(model=model, args=training_args,
                          compute_metrics=compute_metrics,
                          train_dataset=train_encoded,
                          eval_dataset=test_encoded,
                          tokenizer=tokenizer)


        trainer.train()

        preds_output = trainer.predict(test_encoded)
        print(preds_output.metrics)

        # tokenizer.save_pretrained("local-pt-checkpoint")
        model.save_pretrained(args.model_save_path)


    local_model = AutoModelForSequenceClassification.from_pretrained(args.model_save_path)
    local_pipeline = pipeline("text-classification",model=local_model,tokenizer=tokenizer)
    pred = local_pipeline("Wir sind alle auf der Suche nach schnellen Wegen, um fließender Englisch zu sprechen.")
    print(labels_dict_reverse[pred[0]['label'].lstrip("LABEL_")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_and_test", type=bool, default=True)
    parser.add_argument("--model_save_path", type=str, default="local-checkpoint")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)

