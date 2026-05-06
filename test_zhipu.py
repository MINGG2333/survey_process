import os
import base64
import json
from zai import ZhipuAiClient

api_key = os.environ.get("ZHIPU_API_KEY", None)
client = ZhipuAiClient(api_key=api_key)  # 填写您自己的 APIKey

# 读取本地图片并编码为 base64
img_paths = [
    "/mnt/zhitainew/ttt/interview_transcript/survey/1.jpg",
    "/mnt/zhitainew/ttt/interview_transcript/survey/2.jpg"
]

img_bases = []
for img_path in img_paths:
    with open(img_path, "rb") as img_file:
        img_base = base64.b64encode(img_file.read()).decode("utf-8")
        img_bases.append(img_base)

response = client.chat.completions.create(
    model="GLM-4.6V-FlashX",  # 填写需要调用的模型名称
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_bases[0]
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_bases[1]
                    }
                },
                {
                    "type": "text",
                    "text": "请整理为问题答案表格"
                }
            ]
        }
    ],
    thinking={
        "type": "enabled"
    }
)

message = response.choices[0].message
content = getattr(message, "content", message)

output_dir = "glm-output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_zhipu_response.md")

if isinstance(content, list):
    formatted = []
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                formatted.append(block.get("text", ""))
            elif block_type == "image_url":
                formatted.append(f"[image_url: {block.get('image_url', {}).get('url', '')}]")
            else:
                formatted.append(json.dumps(block, ensure_ascii=False, indent=2))
        else:
            formatted.append(str(block))
    md_content = "\n".join(formatted).strip()
elif isinstance(content, dict):
    md_content = json.dumps(content, ensure_ascii=False, indent=2)
else:
    md_content = str(content).strip()

with open(output_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"LLM 回复已保存到 {output_path}")
