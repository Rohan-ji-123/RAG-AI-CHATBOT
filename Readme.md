# how to use the RAG AI Teaching assitance on your own data
## step -1 Collect your videos
Move all your video files to the video folder

## step 2 - Convert to mp3
Convert all the videos files to mp3 by running video_to_mp3

## step 3 - Convert mp3 to json
Convert all the mp3 files to json by running mp3_to_json

## step 4 - Convert the json files to Vector
Use the file process_josn to Convert the json file to a dataframe with Embeddings and save it as a joblib pickle

## step 5 - Prompt genratiom and feeding to LLM
Read the joblib file and load it into the memory. The create a relevent prompt as per the user query and feed it to the LLM