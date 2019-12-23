# DA_Annotation
Model for Dialog Act classification. This model classsify utterances of the  transcript of 'Boston Cookie Theft Description
Task' from Pitt sub corpus of DementiaBank dataset. This task refers to describing conversation between patient 
(Alzheimer Disease or Healthy Control) and an investigator where the investigator shows a picture to the
participant and the participant talks about the picture. The interview is conducted by the investigator as he/she tries to 
elicit more information from the participant.

There are 26 DA tags to annotate the utterance. Each transcript was segmented into utterances by investigator or participant.
The utterances can have multiple labels at the same time. So it is a multi class multi label problem using machine learning model like
Decision Tree, Logistic Regression, SVC, Multi Layer Perceptron. These models classify
each utterance and generate accuracy, jaccard index, micro precision, micro recall, micro F1.
How to Run:
- Run DA_classifier.py