# transformer_finetune_langid

###
The language identification training starts by running 

        python main.py

Set --train_and_test as True to train and save a model using roberta-small
then --train_and_test as True to load the saved model and make a prediction

Experimental logs will be recorded in [WanDB](https://wandb.ai/jgeng/huggingface/runs/didlxof5?workspace=user-jgeng): 
![Experiments](wandb.png)

 We create a fastai application based on onnx model converted from the saved model. Run
 
        uvicorn server:app --reload 

in the fastapi folder.
![FastAPI app](webapp.png)


