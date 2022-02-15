# Explainable Deep Neural Networks for MRI Based Stroke Analysis


Currently, most deep learning models are still black-box methods. Particularly
in medical applications, not only the performance but also the explainability of
artificial intelligence (XAI) is crucial. In this paper, models are examined which are
used to diagnose patients with ischemic strokes. Two different methods of XAI are
used to explain deep neural networks: Grad-Cam and Grad-CAM++. They are used
to show where the class discriminating pixels are located on a Magnetic Resonance
(MR) image. Various experiments show how the explainability of deep learning
models can be improved and whether general statements can be made for the
classification of stroke patients based on their MRIs. Furthermore, a convolutional
neural network (CNN) is implemented to detect an existing stroke and classify
the severity of the disability it causes (mRS Outcome). With an accuracy of 94%,
images of patients can be classified as stroke or no stroke. The XAI shows that both
models can reliably detect brain lesions caused by a stroke. Although the data is
unbalanced, the model that predicts the mRS outcome has an overall accuracy of
92%. It is shown that there are differences in the explanations for mRS Outcome 3-6
and mRS Outcome 0-2. By using Grad-CAM, lesions in the brain can be detected,
which are even overseen by experienced neurologists. In addition, it is possible
to simplify models without significant performance loss by using Grad-CAM. The
resulting explanations can thus serve experts such as neurologists and physicians as
a basis for new hypotheses or even help them to improve their diagnostic quality.
