{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Response API방식"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.라이브러리 가져오고 api key를 환경 변수에서 가져오기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import base64\n",
    "from io import BytesIO\n",
    "\n",
    "from PIL import Image\n",
    "# from dotenv import load_dotenv\n",
    "from openai import OpenAI\n",
    "\n",
    "# load_dotenv()\n",
    "api_key = os.getenv(\"OPENAI_API_KEY\")\n",
    "\n",
    "# client는 보통 전역 1회 생성 권장\n",
    "client = OpenAI(api_key=api_key)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.이미지를 문자열로 인코딩하는 함수 정의하기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 이미지를 base64 문자열로 인코딩하는 함수\n",
    "def encode_image(img: Image.Image, max_side: int = 512) -> str:\n",
    "    w, h = img.size\n",
    "    scale = min(1.0, max_side / max(w, h))\n",
    "    if scale < 1.0:\n",
    "        img = img.resize((int(w * scale), int(h * scale)))\n",
    "\n",
    "    buf = BytesIO()\n",
    "    img.convert(\"RGB\").save(buf, format=\"JPEG\", quality=90)\n",
    "    return base64.b64encode(buf.getvalue()).decode(\"utf-8\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.모델이 이미지 분류 요청 함수 정의하기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# GPT 모델에게 이미지와 프롬프트를 보내 결과를 받아오는 함수\n",
    "def classify_image(prompt: str, img: Image.Image, model: str = \"gpt-4o\") -> str:\n",
    "    b64 = encode_image(img)\n",
    "    data_uri = f\"data:image/jpeg;base64,{b64}\"\n",
    "\n",
    "    resp = client.responses.create(\n",
    "        model=model,\n",
    "        input=[\n",
    "            {\n",
    "                \"role\": \"user\",\n",
    "                \"content\": [\n",
    "                    {\"type\": \"input_text\", \"text\": prompt},\n",
    "                    {\"type\": \"input_image\", \"image_url\": data_uri},\n",
    "                ],\n",
    "            }\n",
    "        ],\n",
    "        temperature=0,\n",
    "    )\n",
    "\n",
    "    return resp.output_text"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.프롬프트 선언하고 이미지 분류 실행하기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\n",
      "    \"building\": 1,\n",
      "    \"sea\": 0,\n",
      "    \"mountain\": 1\n",
      "}\n"
     ]
    }
   ],
   "source": [
    "# GPT에게 보낼 프롬프트 정의\n",
    "prompt = \"\"\"\n",
    "영상을 보고 다음 보기 내용이 포함되면 1, 포함되지 않으면 0으로 분류해줘.\n",
    "보기 = [건축물, 바다, 산]\n",
    "JSON format으로 키는 'building', 'sea', 'mountain'으로 하고 각각 건축물, 바다, 산에 대응되도록 출력해줘.\n",
    "자연 이외의 건축물이 조금이라도 존재하면 'building'을 1로, 물이 조금이라도 존재하면 'sea'을 1로, 산이 조금이라도 보이면 'mountain'을 1로 설정해줘.\n",
    "markdown format은 포함하지 말아줘.\n",
    "\"\"\"\n",
    "\n",
    "img = Image.open('imgs_classification/01.jpg')  # 이미지 열기\n",
    "response = classify_image(prompt, img)     # GPT로부터 분류 결과 받기\n",
    "print(response)  # 결과 출력\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "c:\\Users\\chang\\anaconda3\\envs\\llm_img\\python.exe\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "print(sys.executable)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -0) 라이브러리 추가하기 : streamlit\n",
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-14 10:51:27.765 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:51:27.766 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:51:27.769 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:51:27.771 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:51:27.773 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:51:27.776 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:51:27.779 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:51:27.780 Session state does not function when running a script without `streamlit run`\n",
      "2026-01-14 10:51:27.782 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:51:27.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:51:27.784 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# -1) model 선택하기 : st.sidebar / st.selectbox\n",
    "with st.sidebar:\n",
    "    st.header(\"설정\")\n",
    "    \n",
    "    # API Key가 환경변수에 없으면 입력받기 (선택사항)\n",
    "    api_key = os.getenv(\"OPENAI_API_KEY\")\n",
    "    if not api_key:\n",
    "        api_key = st.text_input(\"OpenAI API Key를 입력하세요\", type=\"password\")\n",
    "    \n",
    "    selected_model = st.selectbox(\n",
    "        \"모델 선택\", \n",
    "        [\"gpt-4o\", \"gpt-4-turbo\", \"gpt-4o-mini\"],\n",
    "        index=0\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-14 10:52:41.336 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:41.338 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:41.340 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:41.342 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:41.345 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:41.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:41.354 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:41.356 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:41.358 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:41.361 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# -2) prompt 작성하기 : st.text_area\n",
    "\n",
    "default_prompt = \"\"\"\n",
    "영상을 보고 다음 보기 내용이 포함되면 1, 포함되지 않으면 0으로 분류해줘.\n",
    "보기 = [건축물, 바다, 산]\n",
    "JSON format으로 키는 'building', 'sea', 'mountain'으로 하고 각각 건축물, 바다, 산에 대응되도록 출력해줘.\n",
    "자연 이외의 건축물이 조금이라도 존재하면 'building'을 1로, 물이 조금이라도 존재하면 'sea'을 1로, 산이 조금이라도 보이면 'mountain'을 1로 설정해줘.\n",
    "markdown format은 포함하지 말아줘.\n",
    "\"\"\"\n",
    "\n",
    "st.subheader(\"1. 프롬프트 작성\")\n",
    "user_prompt = st.text_area(\"이미지 분석을 위한 프롬프트를 입력하세요:\", value=default_prompt.strip(), height=200)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-14 10:52:53.785 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:53.785 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:53.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:53.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:53.789 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:53.791 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:53.792 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:53.793 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-14 10:52:53.794 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# -3) 이미지 업로드하기 : st.file_uploader\n",
    "\n",
    "st.subheader(\"2. 이미지 업로드\")\n",
    "uploaded_file = st.file_uploader(\"분석할 이미지를 선택하세요 (jpg, png, jpeg)\", type=[\"jpg\", \"png\", \"jpeg\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -4) 업로드한 이미지 보여주기 : st.image\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    # PIL Image로 변환\n",
    "    image = Image.open(uploaded_file)\n",
    "    \n",
    "    # 화면을 2분할하여 왼쪽엔 이미지, 오른쪽엔 결과를 보여줌\n",
    "    col1, col2 = st.columns([1, 1])\n",
    "    \n",
    "    with col1:\n",
    "        st.image(image, caption='업로드된 이미지', use_column_width=True)\n",
    "\n",
    "# -5) 분류 실행하기 : st.button /st.spinner\n",
    "    with col2:\n",
    "        st.subheader(\"3. 분석 결과\")\n",
    "        if st.button(\"분류 실행하기\", use_container_width=True):\n",
    "            with st.spinner(\"GPT가 이미지를 분석 중입니다...\"):\n",
    "                try:\n",
    "                    # 함수 호출\n",
    "                    result = classify_image(client, user_prompt, image, selected_model)\n",
    "                    \n",
    "                    # -6) 결과 출력하기 : st.write / st.code\n",
    "                    st.success(\"분석 완료!\")\n",
    "                    st.write(\"결과 (JSON):\")\n",
    "                    st.code(result, language='json')\n",
    "                    \n",
    "                except Exception as e:\n",
    "                    st.error(f\"오류가 발생했습니다: {e}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "llm_img",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
