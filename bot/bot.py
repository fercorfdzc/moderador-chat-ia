import httpx
from twitchio.ext import commands

API_URL = "http://127.0.0.1:8000/predict"

BOT_TOKEN = "oauth:jj1syyde8jxv1eltyapltsl5y9p9yh"  # token del bot con scope moderator:manage:chat_messages
CLIENT_ID = "8r4ltdorvf2pl6i73ewhfpx9kueznp"        # de dev.twitch.tv/console
BROADCASTER_ID = "1452332459"  # ID numérico del canal (no el nombre)
MODERATOR_ID = "1452332459"       # ID numérico del bot
CHANNEL = "ceqw3bot"


class Bot(commands.Bot):

    def __init__(self):
        super().__init__(
            token=BOT_TOKEN,
            prefix="!",
            initial_channels=[CHANNEL]
        )

    async def event_ready(self):
        print(f"Bot conectado como {self.nick}")

    async def delete_message(self, message):
        """Borra un mensaje usando la API REST de Twitch (Helix)"""
        try:
            # El token sin el prefijo "oauth:"
            token = BOT_TOKEN.replace("oauth:", "")

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    "https://api.twitch.tv/helix/moderation/chat",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Client-Id": CLIENT_ID,
                    },
                    params={
                        "broadcaster_id": BROADCASTER_ID,
                        "moderator_id": MODERATOR_ID,
                        "message_id": message.id,
                    }
                )

            if response.status_code == 204:
                print(f"Mensaje de {message.author.name} borrado correctamente.")
            else:
                print(f"Error al borrar: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Error al borrar mensaje: {e}")

    async def event_message(self, message):
        if message.echo:
            return

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    API_URL,
                    json={"message": message.content},
                    timeout=5.0
                )

            result = response.json()

            label = result["label"]
            confidence = result["confidence"]

            print(f"{message.author.name}: {message.content}")
            print(f"Clasificacion: {label} ({confidence:.2f})")

            # Borrar mensaje si es muy subido de tono
            if (label == "hate" or label == "toxic") and confidence > 0.80:
                await self.delete_message(message)
                await message.channel.send(
                    f"@{message.author.name} tu mensaje fue eliminado por lenguaje inapropiado."
                )

            # Detecta hate con confianza media
            elif label == "hate" and confidence > 0.50:
                await message.channel.send(
                    f"@{message.author.name} mensaje detectado como discurso de odio."
                )

            # Detecta tóxico con confianza media
            elif label == "toxic" and confidence > 0.50:
                await message.channel.send(
                    f"@{message.author.name} cuidado con el lenguaje."
                )

        except Exception as e:
            print("Error llamando a la API:", e)

        await self.handle_commands(message)

    @commands.command()
    async def ping(self, ctx):
        await ctx.send("hola")


bot = Bot()
bot.run()