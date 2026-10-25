import os 
import whisper
import json
model = whisper.load_model("base", device="cpu")
audios = os.listdir("audios")

for audio in audios:
    if("_" in audio and audio.lower().endswith(".mp3")):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number,title)
        result = model.transcribe(audio =f"audios/{audio}",
                            language = "hi",
                            task = "translate",
                            word_timestamps = False)
        chunks = []
        for segment in result["segments"]:
            chunks.append({"number" : number, "title" : title, "start" : segment["start"],
                           "end": segment["end"], "text":segment["text"]})

            chunks_with_metadata = {"chunks":chunks , "text":result["text"]}

            with open(f"jsons/{audio.removesuffix('.mp3')}.json", "w") as f:
                json.dump(chunks_with_metadata,f)
