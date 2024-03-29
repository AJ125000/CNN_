CNN Model (10pts)
Implement the CNNClassifier in models.py. Similar to homework 1, your 
model should return a (B,6) torch.Tensor which represents the logits of 
the classes. Use convolutions this time. Use python3 -m grader 
homework -v to grade the first part.

Relevant Operations
• torch.nn.Conv2d


Logging (30pts)
We created a dummy training procedure in acc_logging.py, and provided 
you with two tb.SummaryWriter as logging utilities. Use those summary 
writers to log the training loss at every iteration, the training accuracy at each 
epoch and the validation accuracy at each epoch. Log everything in global 
training steps. Here is a simple example of how to use the SummaryWriter.
import torch.utils.tensorboard as tb
logger = tb.SummaryWriter('cnn')
logger.add_scalar('train/loss', t_loss, 0)
In acc_logging.py, you should not create your own SummaryWriter, but 
rather use the one provided. You can test your logger by calling python3 -m 
homework.acc_logging log, where log is your favorite directory. Then 
start up tensorboard: tensoboard --logdir log. 
Use python3 -m grader homework -v to grade the logging.

Relevant Operations
• torch.utils.tensorboard.SummaryWriter
• torch.utils.tensorboard.SummaryWriter.add_scalar


Training your CNN model (60pts)
Train your model and save it as cnn.th. You can reuse some of the training 
functionality in train.py from homework 1. We highly recommend you 
incorporate the logging functionality from section 2 into your training routine. 
Once you trained your model, you can optionally visualize your model's 
prediction using python3 -m homework.viz_prediction 
[DATASET_PATH].
After implementing everything, you can use python3 -m grader 
homework to test your solutions against the validation grader locally. Your 
model should achieve a 0.85 test accuracy
