import os
import base64
import json
from datetime import datetime
from openai import OpenAI

# 从环境变量中获取您的API KEY
api_key = os.getenv('ARK_API_KEY')

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
)

# 读取本地图片并编码为 base64
img_paths = [
    "/mnt/zhitainew/ttt/interview_transcript/survey/1.jpg",
    "/mnt/zhitainew/ttt/interview_transcript/survey/2.jpg"
]

# 构建 input 内容列表
content_list = []

for img_path in img_paths:
    with open(img_path, "rb") as img_file:
        img_base = base64.b64encode(img_file.read()).decode("utf-8")
    content_list.append({
        "type": "input_image",
        "image_url": f"data:image/jpeg;base64,{img_base}"
    })

content_list.append({
    "type": "input_text",
    "text": "请整理为问题答案表格"
})

# 构造请求 input
request_input = [
    {
        "role": "user",
        "content": content_list,
    }
]

response = client.responses.create(
    model="doubao-seed-1-6-flash-250828",
    input=request_input,
)

# 提取回复文本（LLM 的输出通常是 markdown 格式，直接保存即可）
md_content = ""
for item in response.output:
    if item.type == "message":
        for content_item in item.content:
            if hasattr(content_item, "text"):
                md_content = content_item.text
                break
        break

output_dir = "llm-output"
os.makedirs(output_dir, exist_ok=True)

# 保存 markdown 回复
md_output_path = os.path.join(output_dir, "test_doubao_response.md")
with open(md_output_path, "w", encoding="utf-8") as f:
    f.write(md_content)

# 保存元数据到 json
metadata = {
    "timestamp": datetime.now().isoformat(),
    "model": response.model,
    "input": {
        "messages": request_input,
        "image_paths": img_paths,
    },
    "output": {
        "text_file": md_output_path,
        "text_preview": md_content[:200] + "..." if len(md_content) > 200 else md_content,
    },
    "usage": {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens if response.usage.output_tokens_details else None,
        "total_tokens": response.usage.total_tokens,
    },
    "response_id": response.id,
    "status": response.status,
}

json_output_path = os.path.join(output_dir, "test_doubao_response_meta.json")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"LLM 回复已保存到 {md_output_path}")
print(f"元数据已保存到 {json_output_path}")
print(f"Token 使用情况: 输入={response.usage.input_tokens}, 输出={response.usage.output_tokens}, 总计={response.usage.total_tokens}")
