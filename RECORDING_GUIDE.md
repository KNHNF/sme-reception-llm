# Demo Recording Guide
IGP Viva -- UWE Bristol MSc Data Science

Use ZD Screen Recorder. Record two separate videos.
Keep each under 3 minutes. No editing needed, just clear narration.

---

## Video 1 -- Live Pipeline Demo (2 min)

This shows the system working end-to-end in mock mode.

### Setup before recording

Open two PowerShell windows side by side.
Open ZD Screen Recorder, set to record full screen with microphone audio.

Window 1 -- backend server:
```
cd E:\Coding\public-projects\ai-reception-sme
C:\Users\USER\AppData\Local\Programs\Python\Python311\python.exe -m uvicorn backend:app --reload --port 5005
```
Wait until you see "Application startup complete."

Window 2 -- demo script:
```
cd E:\Coding\public-projects\ai-reception-sme
C:\Users\USER\AppData\Local\Programs\Python\Python311\python.exe demo.py --text --no-tts
```

### What to say on camera

Hit record. Speak clearly. Say something like:

"This is the SME voice assistant pipeline running locally.
The backend is running on port 5005 in mock mode -- no GPU needed.
I'll type a few customer utterances and you can see the pipeline respond."

Then type each of these one at a time, pause between each so the output is visible:

1. `I'd like to book a consultation for next Monday at 2pm`
   -- should return book_appointment, spoken confirmation

2. `Do you have any slots available on Thursday?`
   -- should return check_availability

3. `I need to cancel my appointment on Wednesday at 10am`
   -- should return cancel_appointment

4. `Book me in for a follow-up`
   -- should return clarify (time is missing)

5. `What are your opening hours?`
   -- should return out_of_scope

After each one, briefly say:
"Action is [X], the system responds with [spoken text], latency is [X]ms."

Then open http://localhost:5005/docs in your browser, show the Swagger UI,
click POST /turn, click Try it out, paste one utterance, hit Execute.
Say: "The website team connects to this endpoint directly."

Stop recording.

---

## Video 2 -- Fine-Tuned Model Evidence (2 min)

This shows the trained model results since the fine-tuned model cannot run locally.

### What to show

Open the Kaggle eval notebook output page (you should have the eval run saved).
Or open evaluation/eval_results/eval_summary.json in VS Code.

### What to say

"The fine-tuned models were evaluated on Kaggle using a T4 GPU.
Here are the results comparing vanilla versus fine-tuned for both model families."

Read out the table slowly:

"Vanilla Phi-3: action accuracy 0.4%, JSON valid 90%.
Fine-tuned Phi-3: action accuracy 98.1%, JSON valid 100%.

Vanilla Llama 3: action accuracy 0%, JSON valid 95%.
Fine-tuned Llama 3: action accuracy 99.8%, JSON valid 100%.

Both fine-tuned models went from near-zero to 98-99% accuracy
on a 600-sample synthetic dataset, trained in under 50 minutes on a free Kaggle GPU."

Then open the Kaggle notebook and show the training loss curve if available.
Say: "Training loss dropped from 0.5 at step 10 to 0.028 by step 90,
confirming the model converged correctly."

Stop recording.

---

## Tips

- Record in one take if possible. Mistakes are fine, just keep going.
- Keep the font size large (Ctrl+= in PowerShell or VS Code) so text is readable.
- If something breaks on camera: say "let me restart that" and redo it calmly.
- Total runtime for both videos: under 5 minutes.
- Save as MP4. Upload to UWE OneDrive or share directly with team.

---

## If you want voice input in Video 1

Run this instead of --text mode:
```
python demo.py --no-tts
```
Press Enter, speak clearly into the mic for 5 seconds.
Whisper tiny model transcribes it, pipeline responds.
This is more impressive visually but has a small risk of transcription errors on camera.
Only do this if you have tested it first and it works reliably.
