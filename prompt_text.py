import google.generativeai as genai
import os
import time

genai.configure(api_key="AIzaSyBCgIkI2WP7JE1Qh5IEqkXcxp9r4wbTH70")

model = genai.GenerativeModel('gemini-1.5-flash')


def get_gemini_response(model, prompt, question):
    content = [prompt[0], question]
    task = model.generate_content(content)
    time.sleep(0.5)
    # print(task)
    if task.text == '<REFERRING_EXPRESSION_SEGMENTATION>\n':
        
        target = model.generate_content(['what is the object this user want to inpaint. for example, if the user said "inpaint the white dog with a orange cat", you only need to respond the white dog:',
                                         question])
        time.sleep(0.5)
        object = model.generate_content(['what is the object this user want to inpaint with. for example, if the user said "inpaint the white dog with a orange cat", you only need to respond a orange cat:', 
                                         question])
        return task.text[:-1], target.text[:-1], object.text[:-1]
    elif task.text == '<CAPTION_TO_PHRASE_GROUNDING>\n':
        return [task.text[:-1], question, None]
    else:
        return [task.text[:-1], None, None]
    
    
prompt = ["""
          you are about to solve a semantic question, and you need to classify the following question to a certain task type
          \n
          there are three different task categories: image caption, object detection, and image segmentation.
          \n
          if you think the question is referred to image captioning, your response should be '<DETAILED_CAPTION>' without quotation marks;
          \n
          and if you think the question is referred to object detection, your response should be '<CAPTION_TO_PHRASE_GROUNDING>' without quotation marks;
          \n
          and if you think the question is referred to image segmentation, your response should be '<REFERRING_EXPRESSION_SEGMENTATION>' without quotation marks.
          \n
          for example, if the question is 'describe what is the man doing' or 'what color is the car', you response should be '<DETAILED_CAPTION>' without quotation marks.
          \n
          and if the question is 'locate the man', you response should be '<CAPTION_TO_PHRASE_GROUNDING>' without quotation marks.
          \n
          here is the question:
          """]


# question = 'replace the jeep with a dog'

# print(get_gemini_response(model=model, prompt=prompt, question=question)[0])