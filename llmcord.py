import asyncio
from aiohttp import web
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
import signal
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
CHUNK_TIMEOUT_SECONDS = 30

MAX_MESSAGE_NODES = 500

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information, news, facts, or anything you don't know or are unsure about",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}

SEARCH_TIMEOUT_SECONDS = 30


def get_config() -> dict[str, Any]:
    filename = "config.yaml" if os.path.exists("config.yaml") else "config-example.yaml"
    with open(filename, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if bot_token := os.environ.get("BOT_TOKEN"):
        cfg["bot_token"] = bot_token

    for provider in cfg.get("providers", {}):
        env_key = f"{provider.upper().replace('-', '_')}_API_KEY"
        if api_key := os.environ.get(env_key):
            cfg["providers"][provider]["api_key"] = api_key

    if perplexity_key := os.environ.get("PERPLEXITY_API_KEY"):
        cfg["perplexity_api_key"] = perplexity_key

    return cfg


async def search_web(query: str, httpx_client: httpx.AsyncClient, api_key: str) -> str:
    """Search the web using Perplexity Sonar API."""
    try:
        response = await httpx_client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "sonar-pro",
                "messages": [{"role": "user", "content": query}],
                "search_recency_filter": "month",
                "return_citations": True,
            },
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", [])
        if citations:
            content += "\n\nSources:\n" + "\n".join(f"[{i+1}] {url}" for i, url in enumerate(citations))

        return content
    except Exception as e:
        logging.warning(f"Perplexity search failed for '{query}': {e}")
        return "Search failed. No results available."


async def health_check(request):
    return web.Response(text="OK")


async def start_health_server() -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/", health_check)
    app.router.add_get("/health", health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"Health check server running on port {port}")
    return runner


async def shutdown(runner: Optional[web.AppRunner]) -> None:
    logging.info("Shutting down...")

    if not discord_bot.is_closed():
        try:
            await discord_bot.close()
        except Exception:
            logging.exception("Error closing Discord bot")

    try:
        await httpx_client.aclose()
    except Exception:
        logging.exception("Error closing HTTP client")

    if runner:
        try:
            await runner.cleanup()
        except Exception:
            logging.exception("Error cleaning up health server")


def install_signal_handlers(loop: asyncio.AbstractEventLoop, shutdown_event: asyncio.Event) -> None:
    def request_shutdown(sig: signal.Signals) -> None:
        logging.info("Received %s, shutting down...", sig.name)
        shutdown_event.set()

    def request_shutdown_sync(signum, _frame) -> None:
        try:
            sig = signal.Signals(signum)
            logging.info("Received %s, shutting down...", sig.name)
        except Exception:
            logging.info("Received signal %s, shutting down...", signum)
        loop.call_soon_threadsafe(shutdown_event.set)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, request_shutdown, sig)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, request_shutdown_sync)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(
        model=model,
        messages=messages[::-1],
        stream=True,
        tools=[SEARCH_TOOL],
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body
    )

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    async def update_embed(description: str, color=EMBED_COLOR_INCOMPLETE) -> None:
        embed.description = description
        embed.color = color
        if response_msgs:
            await response_msgs[-1].edit(embed=embed)

    async def execute_tool_call(tc: dict) -> dict:
        try:
            args = json.loads(tc["arguments"])
            query = args.get("query", "")
        except json.JSONDecodeError:
            query = ""

        truncated_query = f'{query[:50]}{"..." if len(query) > 50 else ""}'
        if not use_plain_responses:
            await update_embed(f'🔍 Searching: "{truncated_query}"')

        perplexity_key = config.get("perplexity_api_key", "")
        result = await search_web(query, httpx_client, perplexity_key)

        if not use_plain_responses:
            await update_embed("📊 Analyzing results...")

        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result
        }

    def accumulate_tool_call(tool_calls_buffer: dict, tc) -> None:
        idx = tc.index
        if idx not in tool_calls_buffer:
            tool_calls_buffer[idx] = {"id": "", "name": "", "arguments": ""}
        if tc.id:
            tool_calls_buffer[idx]["id"] = tc.id
        if tc.function and tc.function.name:
            tool_calls_buffer[idx]["name"] = tc.function.name
        if tc.function and tc.function.arguments:
            tool_calls_buffer[idx]["arguments"] += tc.function.arguments

    async def stream_content(choice, search_count: int) -> None:
        global last_task_time
        nonlocal curr_content

        prev_content = curr_content or ""
        curr_content = choice.delta.content or ""
        finish_reason = choice.finish_reason

        new_content = prev_content if finish_reason is None else (prev_content + curr_content)

        if not response_contents and not new_content:
            return

        start_next_msg = not response_contents or len(response_contents[-1] + new_content) > max_message_length
        if start_next_msg:
            response_contents.append("")

        response_contents[-1] += new_content

        if use_plain_responses or not response_contents[-1].strip():
            return

        time_delta = datetime.now().timestamp() - last_task_time
        ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
        msg_split_incoming = finish_reason is None and len(response_contents[-1] + curr_content) > max_message_length
        is_final_edit = finish_reason is not None or msg_split_incoming
        is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")

        if not (start_next_msg or ready_to_edit or is_final_edit):
            return

        if search_count > 0:
            embed.set_footer(text="🔍" if search_count == 1 else f"🔍 ×{search_count}")

        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

        if start_next_msg:
            await reply_helper(embed=embed, silent=True)
        else:
            await asyncio.sleep(max(0, EDIT_DELAY_SECONDS - time_delta))
            await response_msgs[-1].edit(embed=embed)

        last_task_time = datetime.now().timestamp()

    search_count = 0

    try:
        async with new_msg.channel.typing():
            api_messages = openai_kwargs.pop("messages")

            while True:
                tool_calls_buffer = {}
                finish_reason = None
                stream = await openai_client.chat.completions.create(messages=api_messages, **openai_kwargs)

                while True:
                    try:
                        chunk = await asyncio.wait_for(stream.__anext__(), timeout=CHUNK_TIMEOUT_SECONDS)
                    except (StopAsyncIteration, asyncio.TimeoutError):
                        break

                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue

                    finish_reason = choice.finish_reason

                    if choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            accumulate_tool_call(tool_calls_buffer, tc)
                        if not use_plain_responses and not response_msgs:
                            embed.description = "🔍 Searching..."
                            embed.color = EMBED_COLOR_INCOMPLETE
                            await reply_helper(embed=embed, silent=True)
                            response_contents.append("")

                    elif choice.delta.content or finish_reason:
                        await stream_content(choice, search_count)

                    if finish_reason and finish_reason.lower() in ("stop", "end_turn"):
                        break

                if finish_reason != "tool_calls" or not tool_calls_buffer:
                    break

                if response_contents:
                    response_contents[-1] = ""

                assistant_tool_calls = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in (tool_calls_buffer[idx] for idx in sorted(tool_calls_buffer.keys()))
                ]

                tool_results = []
                for idx in sorted(tool_calls_buffer.keys()):
                    tc = tool_calls_buffer[idx]
                    if tc["name"] == "search_web":
                        tool_results.append(await execute_tool_call(tc))
                        search_count += 1

                api_messages.append({"role": "assistant", "tool_calls": assistant_tool_calls})
                api_messages.extend(tool_results)
                curr_content = None

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")

    finally:
        if not use_plain_responses and response_msgs and response_contents and response_contents[-1].strip():
            is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")
            embed.description = response_contents[-1]
            embed.color = EMBED_COLOR_COMPLETE if is_good_finish else EMBED_COLOR_INCOMPLETE
            if search_count > 0:
                embed.set_footer(text="🔍" if search_count == 1 else f"🔍 ×{search_count}")
            try:
                await response_msgs[-1].edit(embed=embed)
            except Exception:
                logging.exception("Error during final embed edit")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    runner = await start_health_server()
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    install_signal_handlers(loop, shutdown_event)

    bot_task = asyncio.create_task(discord_bot.start(config["bot_token"]), name="discord-bot")
    shutdown_task = asyncio.create_task(shutdown_event.wait(), name="shutdown-wait")

    await asyncio.wait(
        {bot_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    await shutdown(runner)

    shutdown_task.cancel()
    await asyncio.gather(shutdown_task, return_exceptions=True)
    await asyncio.gather(bot_task, return_exceptions=True)

    if bot_task.done():
        exc = bot_task.exception()
        if exc:
            raise exc


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
