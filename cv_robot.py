import os
import cv2
import base64
import time
import asyncio
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

ROBOT_ID = os.environ.get("ROBOT_ID", "Robot_CV")
HF_TOKEN = os.environ.get("HF_CV_ROBOT_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Check .env file.")

MCP_SERVER_URL = "https://mcp-1st-birthday-cv-mcp-server.hf.space/gradio_api/mcp/"
SERVER_NAME = "CV_MCP_Server"
TOOL_NAME = "CV_MCP_Server_robot_watch"

console = Console()

# -------------------------------
# Initialize MCP Client
# -------------------------------
HTTP_TRANSPORT = StreamableHttpTransport(url=MCP_SERVER_URL)
MCP_CLIENT = Client(transport=HTTP_TRANSPORT, name=SERVER_NAME)


def pretty_print_response(resp: dict):
    """Rich table output with row lines, no URL."""
    table = Table(
        title="😎 Robot Vision Result",
        title_style="bold cyan",
        title_justify="left",
        box=box.ROUNDED,
        show_lines=True,
        show_header=False,
        style="bold magenta"
    )

    objects_list = resp.get("objects", [])
    objects_str = ", ".join(objects_list) if isinstance(objects_list, list) else str(objects_list)

    table.add_column("Field", style="bold magenta")
    table.add_column("Value", style="white")

    table.add_row("🤖 Robot ID", str(resp.get("robot_id", "N/A")))
    table.add_row("🏞️  Image Size", str(resp.get("file_size_bytes", "N/A")))
    table.add_row("📝 Description", str(resp.get("description", "N/A")))
    table.add_row("🏛️  Environment", str(resp.get("environment", "N/A")))
    table.add_row("🚪 Indoors/Outdoors", str(resp.get("indoor_or_outdoor", "N/A")))
    table.add_row("💡 Lighting Condition", str(resp.get("lighting_condition", "N/A")))
    table.add_row("👥 Human", str(resp.get("human", "N/A")))
    table.add_row("🐶 Animals", str(resp.get("animals", "N/A")))
    table.add_row("📦 Objects", objects_str)
    table.add_row("🚧 Hazards", str(resp.get("hazards", "N/A")))


    console.print(table)


async def send_frame_to_mcp(b64_img: str):
    """
    Send the Base64 image to MCP tool asynchronously and return structured response.
    Includes both top-level fields (robot_id, file_size_bytes) and 'result' contents.
    """
    payload = {
        "hf_token_input": HF_TOKEN,
        "robot_id_input": ROBOT_ID,
        "image_b64_input": b64_img
    }

    try:
        async with MCP_CLIENT:
            response = await MCP_CLIENT.call_tool(TOOL_NAME, payload)

            if response.is_error:
                error_text = response.content[0].text if response.content and isinstance(response.content, list) else "Unknown error"
                raise Exception(f"MCP Tool Error: {error_text}")

            import ast
            raw_text = response.content[0].text
            resp_dict = ast.literal_eval(raw_text)  # Convert Python string to dict

            # Extract top-level fields
            robot_id = resp_dict.get("robot_id", "N/A")
            file_size = resp_dict.get("file_size_bytes", 0)

            # Extract 'result' subfields
            result = resp_dict.get("result", {})
            description = result.get("description", "")
            environment = result.get("environment", "")
            indoor_outdoor = result.get("indoor_or_outdoor", "")
            lighting = result.get("lighting_condition", "")
            human = result.get("human", "")
            animals = result.get("animals", "")
            objects_list = result.get("objects", [])
            hazards = result.get("hazards", "")

            return {
                "robot_id": robot_id,
                "file_size_bytes": file_size,
                "description": description,
                "environment": environment,
                "indoor_or_outdoor": indoor_outdoor,
                "lighting_condition": lighting,
                "human": human,
                "animals": animals,
                "objects": objects_list,
                "hazards": hazards
            }

    except Exception as e:
        console.print(f"[bold red]Error calling MCP Tool:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return {}


def start_stream():
    """
    Capture frames from webcam and send them to MCP server for VLM analysis.
    """
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        console.print("[bold red]Camera not opened.[/bold red]")
        return

    console.print("[bold green]Camera streaming... Press Ctrl+C to stop.[/bold green]")

    async def stream_loop():
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Encode frame as JPEG and Base64
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            b64_img = base64.b64encode(jpeg.tobytes()).decode("utf-8")

            # Send to MCP server
            resp = await send_frame_to_mcp(b64_img)
            pretty_print_response(resp)

            await asyncio.sleep(1)  # Delay between frames

    try:
        asyncio.run(stream_loop())
    except KeyboardInterrupt:
        console.print("[bold yellow]Streaming stopped by user.[/bold yellow]")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_stream()
