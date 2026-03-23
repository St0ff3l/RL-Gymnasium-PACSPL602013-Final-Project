import base64
import json
from io import BytesIO
from urllib import request

import imageio.v2 as imageio


DEFAULT_PROMPT = (
	"Observe the car. Is it on the LEFT slope, RIGHT slope, or BOTTOM? "
	"Reply with only one word."
)


def query_llava_position(
	frame,
	model: str = "llava:7b",
	prompt: str = DEFAULT_PROMPT,
	host: str = "http://localhost:11434",
	timeout: int = 120,
) -> str:
	"""Send one environment frame to local Ollama LLaVA and return its text reply."""
	buffer = BytesIO()
	imageio.imwrite(buffer, frame, format="png")
	image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

	payload = {
		"model": model,
		"prompt": prompt,
		"stream": False,
		"images": [image_b64],
	}
	req = request.Request(
		url=f"{host.rstrip('/')}/api/generate",
		data=json.dumps(payload).encode("utf-8"),
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	with request.urlopen(req, timeout=timeout) as response:
		data = json.loads(response.read().decode("utf-8"))
	return data.get("response", "").strip()
