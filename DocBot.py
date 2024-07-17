import streamlit as st
from PyPDF2 import PdfReader
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import os
import google.generativeai as genai
from PIL import Image, ImageDraw,ImageEnhance,PngImagePlugin
import numpy as np
import cv2
import fitz
import io
from datetime import datetime

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
genai.configure(api_key='AIzaSyDMAqL5ga6BQzk_UJwmahsFuSNz4Awm-5c')
model_gem_pro = genai.GenerativeModel('gemini-pro')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style_docbot.css")

if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = []
if 'current_chat_messages' not in st.session_state:
    st.session_state.current_chat_messages = []
if 'all_chat_messages' not in st.session_state:
    st.session_state.all_chat_messages = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = []
if 'chat_stream_titles' not in st.session_state:
    st.session_state.chat_stream_titles = []
if 'chat_under_consideration_index' not in st.session_state:
    st.session_state.chat_under_consideration_index = 0

def submit_pdfs():
    if(len(files)):
        st.session_state.submitted=True
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",)
        llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"temperature":0.55, "max_length":10}
            )
        for i in range(len(files)):
            process_pdf(text_splitter,docs,files[i],i)
        st.session_state.faiss_index = FAISS.from_documents(docs, instructor_embeddings)
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            st.session_state.faiss_index.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True,
        )
    
def process_pdf(text_splitter,docs,pdf_file,file_index):
    pdf_reader = PdfReader(pdf_file)
    
    split_text = ""

    for i in range(len(pdf_reader.pages)):
        # print(pdf_reader.pages[i].extract_text())
        split_text = text_splitter.split_text(pdf_reader.pages[i].extract_text())
        
        for j in split_text:
            doc = Document(page_content=j, metadata={"index": i + 1,"file_index":file_index})
            docs.append(doc)

def new_chat_maker():
    if st.session_state.current_chat_messages!=[] and st.session_state.chat_under_consideration_index == len(st.session_state.all_chat_messages):
        st.session_state.all_chat_messages.append(st.session_state.current_chat_messages)
        title_suggestions = model_gem_pro.generate_content("Give an appropriate title/summary in less than 5 words for the answer, I'll directly show your response to user, if you can't provide a title just return the truncated version of the query "+st.session_state.current_chat_messages[0][1])
        title_suggestions.resolve()
        st.session_state.chat_stream_titles.append(title_suggestions.text)
        st.session_state.current_chat_messages = []
        st.session_state.chat_under_consideration_index+=1
        print(st.session_state.chat_under_consideration_index)
        change_messages()
    elif st.session_state.current_chat_messages!=[] and st.session_state.chat_under_consideration_index != len(st.session_state.all_chat_messages):
        st.session_state.all_chat_messages[st.session_state.chat_under_consideration_index] = st.session_state.current_chat_messages
        st.session_state.chat_under_consideration_index = len(st.session_state.all_chat_messages)
        st.session_state.current_chat_messages = []
        change_messages()
   
def create_text_image(page):
    mat = fitz.Matrix(2, 2)
    page_image = page.get_pixmap(matrix=mat)
    th = int(page_image.height * 0.05)
    chth = page_image.height * 0.01
    cwth = page_image.width * 0.01
    cath = chth * cwth * 100
    text_image = Image.new('RGB', (page_image.width, page_image.height), (0, 0, 0))
    draw = ImageDraw.Draw(text_image)
    blocks = page.get_text("blocks")
    for block in blocks:
        x, y, w, h, text, _, _ = block
        draw.rectangle([2*x, 2*y, 2*w, 2*h], fill="white")
    maskArray = np.array(text_image)
    image = cv2.cvtColor(maskArray, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append((x, y, w, h))
    diagArray = np.maximum(maskArray, np.frombuffer(page_image.samples, dtype=np.uint8).reshape(page_image.h, page_image.w, page_image.n))
    diagGray = cv2.cvtColor(diagArray, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diagGray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < cwth or h < chth or w*h < cath:
            continue
        y -= th//2
        h += th
        temp = []
        for block in rectangles:
            x1 = int(block[0]) - th//2
            y1 = int(block[1]) - th//2
            w1 = int(block[2]) + th
            h1 = int(block[3]) + th
            if ((y1 < y and y1+h1 > y) or (y < y1 and y+h > y1)) and (((x1 < x and x1+w1 > x) or (x < x1 and x+w > x1))):
                temp.append((x1, y1, w1, h1))
        temp.append((x, y, w, h))
        selected.append(temp)
    maskArray[0:page_image.height, 0:page_image.width] = 255
    for sel in selected:
        for i, temp in enumerate(sel):
            x, y, w, h = temp
            maskArray[y: y + h, x: x + w] = 0
    diagArray = np.maximum(maskArray, np.frombuffer(page_image.samples, dtype=np.uint8).reshape(page_image.h, page_image.w, page_image.n))
    diagram = Image.fromarray(diagArray)
    ddraw = ImageDraw.Draw(diagram)
    images = []
    for i, sel in enumerate(selected):
        if len(sel) == 0:
            continue
        imx, imy, imw, imh = sel[0]
        for temp in sel:
            x, y, w, h = temp
            imx = min(imx, x)
            imy = min(imy, y)
        for i, temp in enumerate(sel):
            x, y, w, h = temp
            imw = max(imw + imx, w + x) - imx
            imh = max(imh + imy, h + y) - imy
        new = diagArray[imy:imy + imh, imx:imx + imw]
        images.append(new)
    return images

def image_extractor(page,response):
    
    images = create_text_image(page)
    for i in range(len(images)):
        model_gem_pro_vis = genai.GenerativeModel('gemini-1.5-flash')
        try:
            image_relevance = model_gem_pro_vis.generate_content(["On a scale of 1 to 10 (only give the number as response), How much relevant is the image provided to the text - " + response,Image.fromarray(images[i])])
            image_relevance.resolve()
            print(image_relevance.text)
            try:
                if int(image_relevance.text) > 7:
                    return [images[i]]
            except:
                continue
        except:
            continue
    

    return []
    
def query_response(prompt):
    result = st.session_state.qa_chain({'question': prompt,'chat_history':st.session_state.current_chat_messages})
    doc_page_no = []
    image_to_return = []
    
    for i in result['source_documents']:
        pdf_file = PdfReader(files[i.metadata['file_index']])
        doc_page_no.append([i.metadata['file_index'],i.metadata['index']])

        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(pdf_file.pages[i.metadata['index']-1])

        
        pdf_bytes_io = io.BytesIO()
        pdf_writer.write(pdf_bytes_io)
        pdf_bytes_io.seek(0)  

        
        pdf_document = fitz.open(stream=pdf_bytes_io, filetype="pdf")

        
        fitz_page = pdf_document[0]






        image_to_return = image_extractor(fitz_page,result['answer'][result['answer'].rfind('Helpful Answer:'):][16:])
        if image_to_return != []:
            break

    response = '''\nMost Relevant pages:\n'''
    for i in doc_page_no[:3]:
        if files[i[0]].name + '-' + str(i[1]) + '\n' not in response:
            response += '"'+files[i[0]].name+'"' + '-' + str(i[1]) + '\n'


    response = result['answer'][result['answer'].rfind('Helpful Answer:'):][16:] + response



    return [response,image_to_return]

def change_messages():
    # print("in change messages" ,st.session_state.current_chat_messages)
    for i in st.session_state.current_chat_messages:
        # print(i)
        messages_container.chat_message("user").write(i[0]) 
        messages_container.chat_message("assistant").write(i[1])

def retrieve_chat(index):
    new_chat_maker()
    st.session_state.chat_under_consideration_index = index
    print(st.session_state.chat_under_consideration_index)
    st.session_state.current_chat_messages = st.session_state.all_chat_messages[index]
    change_messages()
    
st.markdown("<h1 style='text-align: center;color:#e61542;'>DOCBOT</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; margin:5px'>YAMAHA X IIT MANDI HACKATHON</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: center; color:#e61542;'>Instructions</h1>", unsafe_allow_html=True)
st.sidebar.write(
    '''1. Upload the PDF file(s) using the file uploader
2. Click Submit
3. Query information from the provided texts. A relevant images (if exists) will also be returned.
4. If you want to ask a question without the previous chat history, click the button below (after providing PDF)
5. You can also access previous chats
'''
)
st.sidebar.button('New chat',disabled = not st.session_state.submitted,on_click = new_chat_maker)
st.sidebar.markdown("<h1 style='text-align: center; color:#e61542;'>Previous Chats</h1>", unsafe_allow_html=True)
files = st.file_uploader("Input PDF file(s):", accept_multiple_files=True,disabled=st.session_state.submitted, type="pdf")
submit_button = st.button("Submit",on_click = submit_pdfs,disabled=st.session_state.submitted)
if submit_button:
     st.session_state.submitted = True
for chat_stream_title_index in range(len(st.session_state.chat_stream_titles)-1,-1,-1):
    st.sidebar.button(st.session_state.chat_stream_titles[chat_stream_title_index],key = chat_stream_title_index,on_click = retrieve_chat, args=(chat_stream_title_index,))
messages_container = st.container()
if prompt := st.chat_input("Say something",disabled=not st.session_state.submitted):
    response = query_response(prompt)
    st.session_state.current_chat_messages.append((prompt,response[0]))
    change_messages()
    
    try:
        st.image(response[1][0])
    except:
        pass
