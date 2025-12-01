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
load_dotenv()  # Load variables from .env file

# Robot and Hugging Face token configuration
ROBOT_ID = os.environ.get("ROBOT_ID", "Robot_CV")  # Default robot ID if not set
HF_TOKEN = os.environ.get("HF_CV_ROBOT_TOKEN")  # Hugging Face token for MCP server
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Check .env file.")  # Ensure token is set

# MCP server configuration
MCP_SERVER_URL = "https://mcp-1st-birthday-cv-mcp-server.hf.space/gradio_api/mcp/"
SERVER_NAME = "CV_MCP_Server"  # Name for MCP client
TOOL_NAME = "CV_MCP_Server_robot_watch"  # Tool name on MCP server

console = Console()  # Rich console for pretty printing

# -------------------------------
# Initialize MCP Client
# -------------------------------
HTTP_TRANSPORT = StreamableHttpTransport(url=MCP_SERVER_URL)  # Transport layer
MCP_CLIENT = Client(transport=HTTP_TRANSPORT, name=SERVER_NAME)  # MCP client instance


def pretty_print_response(resp: dict):
    """
    Display MCP server response in a formatted table using Rich.
    Shows all detected fields including humans, animals, objects, hazards, and environment info.
    """
    table = Table(
        title="😎 Robot Vision Result",
        title_style="bold cyan",
        title_justify="left",
        box=box.ROUNDED,
        show_lines=True,
        show_header=False,
        style="bold magenta"
    )

    # Format objects list as comma-separated string
    objects_list = resp.get("objects", [])
    objects_str = ", ".join(objects_list) if isinstance(objects_list, list) else str(objects_list)

    # Add table columns
    table.add_column("Field", style="bold magenta")
    table.add_column("Value", style="white")

    # Add rows for each field
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

    console.print(table)  # Print table to console


async def send_frame_to_mcp(b64_img: str):
    """
    Send a Base64-encoded image to the MCP server asynchronously.
    Returns a structured dictionary with scene analysis:
      - robot_id, file size, description, environment, lighting, humans, animals, objects, hazards
    """
    payload = {
        "hf_token_input": HF_TOKEN,  # Authentication token
        "robot_id_input": ROBOT_ID,  # Robot ID for tracking
        "image_b64_input": b64_img  # Encoded image
    }

    try:
        async with MCP_CLIENT:  # Open MCP client session
            response = await MCP_CLIENT.call_tool(TOOL_NAME, payload)  # Call the MCP tool

            # Handle errors
            if response.is_error:
                error_text = response.content[0].text if response.content and isinstance(response.content, list) else "Unknown error"
                raise Exception(f"MCP Tool Error: {error_text}")

            # Convert response text to dictionary
            import ast
            raw_text = response.content[0].text
            resp_dict = ast.literal_eval(raw_text)

            # Extract top-level fields
            robot_id = resp_dict.get("robot_id", "N/A")
            file_size = resp_dict.get("file_size_bytes", 0)

            # Extract detailed 'result' fields
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
        return {}  # Return empty dict on error


def start_stream():
    """
    Capture frames from a webcam and send them to the MCP server for real-time VLM analysis.
    Displays results in console as a Rich table.
    """
    cap = cv2.VideoCapture(2)  # Open camera (device 2)
    if not cap.isOpened():
        console.print("[bold red]Camera not opened.[/bold red]")
        return

    console.print("[bold green]Camera streaming... Press Ctrl+C to stop.[/bold green]")

    async def stream_loop():
        while True:
            ret, frame = cap.read()  # Capture frame
            if not ret:
                continue

            # Encode frame as JPEG and convert to Base64
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            b64_img = base64.b64encode(jpeg.tobytes()).decode("utf-8")

            # Send frame to MCP server and get response
            resp = await send_frame_to_mcp(b64_img)
            pretty_print_response(resp)  # Display results

            await asyncio.sleep(1)  # Wait before sending next frame

    try:
        asyncio.run(stream_loop())  # Run asynchronous streaming loop
    except KeyboardInterrupt:
        console.print("[bold yellow]Streaming stopped by user.[/bold yellow]")
    finally:
        cap.release()  # Release camera
        cv2.destroyAllWindows()  # Close any OpenCV windows


if __name__ == "__main__":
    start_stream()  # Start the webcam streaming
