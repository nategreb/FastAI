from fastai.vision.all import *
import gradio as gr

learn = load_learner('export.pkl')
labels = learn.dls.vocab


def predict(img):
    #img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


# images from deep fake or real data set: 256 x 256
gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(256, 256)), outputs=gr.outputs.Label(num_top_classes=3)).launch(
    share=True)
