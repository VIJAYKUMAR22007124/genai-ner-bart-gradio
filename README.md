## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Named Entity Recognition (NER) is a key task in Natural Language Processing that involves identifying and classifying entities such as names, locations, organizations, and dates within text. However, deploying high-performing NER models for real-time use remains challenging due to complexities in model fine-tuning and accessibility. This project aims to fine-tune a BART model for accurate entity recognition and develop an interactive Gradio-based interface, enabling users to input text and instantly view detected entities with ease and clarity.

### DESIGN STEPS:

#### STEP 1: Model Setup and Preprocessing

1. Import required libraries and Hugging Face Transformers.  
2. Load pre-trained models for NLP tasks like NER, summarization, or sentiment analysis.  
3. Preprocess and tokenize the input text for model compatibility.  


#### STEP 2: Building Interactive Interface with Gradio

1. Define Python functions to handle model predictions (e.g., `ner()` for entity extraction).  
2. Create a Gradio interface with text input and output components.  
3. Add example sentences to test and visualize model results easily. 


#### STEP 3: Evaluation and Deployment

1. Test the model through the Gradio interface using sample inputs.  
2. Evaluate the performance based on accuracy and entity detection quality.  
3. Deploy the app on Hugging Face Spaces or run it in Google Colab for public use.


### PROGRAM:

```
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # Merge continuation tokens
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["My name is Vijay, I live in Chennai and work at Google.","Elon Musk founded SpaceX in 2002.","Vijay B completed his project at Anna University in 2025.","Barack Obama was born in Hawaii.","Dr. Vijay visited IIT Madras on October 10, 2025."])

demo.launch(share=True, server_port=int(os.environ['PORT4']))
```

### OUTPUT:

<img width="937" height="383" alt="image" src="https://github.com/user-attachments/assets/e4deba3c-1499-4561-b1c1-98217acfc900" />

### RESULT:
The developed prototype successfully performs Named Entity Recognition (NER) using a fine-tuned BART model integrated with the Gradio framework. The application accurately identifies and highlights entities such as persons, organizations, locations, and dates from user-entered text. The interactive Gradio interface provides a simple and effective platform for testing and evaluating model performance, demonstrating the potential of transformer-based models like BART in real-time NER applications.

