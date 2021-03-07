from pytube import YouTube

yt = YouTube("https://www.youtube.com/watch?v=nIwU-9ZTTJc").streams.first().download()
