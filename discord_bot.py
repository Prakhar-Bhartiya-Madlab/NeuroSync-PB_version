# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

import discord
from discord.ext import commands
import requests
import json
import os
from pydub import AudioSegment  # Add this import

# Add message content intent to the list of intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize the bot with intents
bot = commands.Bot(command_prefix='!', intents=intents)

# Define a command to handle audio file uploads
@bot.command()
async def upload_audio(ctx):
    # Check if the message has an attachment
    if len(ctx.message.attachments) == 0:
        await ctx.send('Please upload an audio file with this command.')
        return

    attachment = ctx.message.attachments[0]

    # Check the file size (Discord reports size in bytes)
    if attachment.size > 20 * 1024 * 1024:  # 20 MB limit
        await ctx.send('File size exceeds 20 MB limit. Please upload a smaller file.')
        return

    # Download the attachment
    audio_data = await attachment.read()

    # Check if the attachment is an MP3 file
    if attachment.filename.endswith('.mp3'):
        # Convert MP3 to WAV
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        audio_segment = audio_segment.set_frame_rate(88200)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        audio_data = wav_io.getvalue()
    elif not attachment.filename.endswith('.wav'):
        await ctx.send('Please upload a WAV or MP3 audio file.')
        return

    # Send the audio data to the blendshape API
    blendshape_api_url = "http://YOUR-IP-ADDRESS:7777/audio_to_blendshapes"
    r = requests.post(blendshape_api_url, data=audio_data)
    if r.status_code == 200:
        blendshape_data = r.json()

        # Save blendshape data to a text file
        with open('blendshapes.txt', 'w') as f:
            json.dump(blendshape_data, f, indent=4)

        # Send the text file back to Discord channel
        await ctx.send(file=discord.File('blendshapes.txt'))
    else:
        await ctx.send('Failed to generate blendshapes.')

# Run the bot
bot.run('YOURAPITHINGYANDSTUFF')
