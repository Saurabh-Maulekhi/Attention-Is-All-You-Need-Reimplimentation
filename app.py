from translation import translation
import gradio as gr


## Documentaion function
def documentation():
    html = """
    <style>
            #docs ul {
                list-style-position: inside; /* Aligns bullets correctly */
                padding-left: 20px;
            }

            #docs li {
                margin-bottom: 10px; /* Adds spacing between items */
            }

            #docs a {
                display: inline-block; /* Prevents full-width issue */
                font-size: 20px;
                width : 600px;
                border-radius: 10px;
                color: white;
                text-decoration: none;
                padding: 5px 10px;
            }

            #docs a:hover {
                color: white;
                background-color: #F16924;
            }
     </style>

        <nav id="docs">
            <ul>
                <li><a href="https://www.linkedin.com/in/saurabh-maulekhi-326584241" target="_blank" rel="noopener noreferrer"><b>Linkedin Id</b></a></li>
                <li><a href="https://www.kaggle.com/code/saurabhmaulekhi/attention-is-all-you-need-paper-reimplimentation" target="_blank" rel="noopener noreferrer"><b>Attention Is All You Need Paper reimplimenatation and Model Deployment (Kaggle Notebook)</b></a></li>
                <li><a href="https://www.kaggle.com/models/saurabhmaulekhi/english-to-hindi-translation" target="_blank" rel="noopener noreferrer"><b>Model Download (Kaggle Space)</b></a></li>
                <li><a href="https://www.linkedin.com/posts/saurabh-maulekhi-326584241_notes-on-atention-is-all-you-need-paper-activity-7312151373633003521-KLcZ?utm_source=share&utm_medium=member_desktop&rcm=ACoAADwQMM4BXaCYT_LwYhxeLC7BWY3KrAVFQOY" target="_blank" rel="noopener noreferrer"><b>Attention is all You Need (Transformers) Paper in depth Notes(Kaggle Notebook)</b></a></li>
                <li><a href="https://github.com/Saurabh-Maulekhi/Attention-Is-All-You-Need-Reimplimentation" target="_blank" rel="noopener noreferrer"><b>Github Repo of This Web App</b></a></li>
                <li><a href="#" target="_blank" rel="noopener noreferrer"><b>Explain/Demo video</b></a></li>
            </ul>
        </nav>
    """
    return html


## Gradio app
examples =  ["Hello my friend how are you ?", "It is a very beautiful day"]

with gr.Blocks() as app:
    gr.Markdown("# Attention is all You Need Paper Reimplimentation")
    gr.Markdown("## English to Hindi Translation")

    with gr.Tab("English-Hindi translation"):
        user_input = gr.Textbox( label = "Your's Input:", placeholder="Your English Sentence")
        submit_button = gr.Button("Submit")

        output = gr.Textbox(label = "Translation:", placeholder="Your translated Hindi Sentence")

        submit_button.click(fn=translation,
                            inputs=[user_input],
                            outputs=output)

        # Adding examples
        gr.Examples(
            examples=examples,
            inputs=user_input,
            outputs=output,
            fn=translation  # This ensures auto-submit on example click
        )

    with gr.Tab("Documentation"):
        gr.HTML(documentation())

app.launch(debug=True)
