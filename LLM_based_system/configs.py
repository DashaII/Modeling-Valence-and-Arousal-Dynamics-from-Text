from pathlib import Path


# OpenAI models
openai_key = "INSERT YOUR OPENAI API KEY HERE"

model_openai_gpt_5_nano = "gpt-5-nano-2025-08-07"
model_openai_gpt_5 = "gpt-5-2025-08-07"
model_openai_gpt_5_1 = "gpt-5.1-2025-11-13"

# Ollama models
ollama_url = "INSERT_OLLAMA_URL"
model_qwen_2_5_72b = "qwen2.5:72b"
model_gpt_oss_120b = "gpt-oss:120b"

# Initial files provided by SemEval Task 2
raw_train_data = r"data\TRAIN_RELEASE_3SEP2025\TRAIN_RELEASE_3SEP2025\train_subtask1.csv"
raw_test_data = r"data\TEST_RELEASE_5JAN2026\test_subtask1.csv"
# BASE_DIR = Path(__file__).resolve().parent
# raw_test_data = BASE_DIR / "data" / "TEST_RELEASE_5JAN2026" / "test_subtask1.csv"

# Initial files enriched with emotions
train_data_emo = r"data\train_data_emotion.csv"
test_data_emo = r"data\test_subtask1_enriched.csv"

# Dictionary from Valence-Arousal to Emotion grid
va_to_emotions = {
    (-2,2): "Jittery, nervous",
    (-1,2): "Somewhat jittery",
    (0,2): "Active",
    (1,2): "Somewhat lively",
    (2,2): "Lively, enthusiastic",
    (-2,1): "Very sad",
    (-1,1): "Somewhat sad",
    (0,1): "Neutral",
    (1,1): "Somewhat happy",
    (2,1): "Very happy",
    (-2,0): "Sluggish, tired",
    (-1,0): "Somewhat sluggish",
    (0,0): "Quiet",
    (1,0): "Somewhat content",
    (2,0): "Content, calm"
}

# Dictionary from Emotion grid to Valence-Arousal
emotions_to_va = {
    "Jittery, nervous": (-2,2),
    "Somewhat jittery": (-1,2),
    "Active": (0,2),
    "Somewhat lively": (1,2),
    "Lively, enthusiastic": (2,2),
    "Very sad": (-2,1),
    "Somewhat sad": (-1,1),
    "Neutral": (0,1),
    "Somewhat happy": (1,1),
    "Very happy": (2,1),
    "Sluggish, tired": (-2,0),
    "Somewhat sluggish": (-1,0),
    "Quiet": (0,0),
    "Somewhat content": (1,0),
    "Content, calm": (2,0)
}

# Max distance from emotion - furthest possible located emotion in comp with the main emotion (v,a)
emotions_to_va_contra = {
    (-2,2): (2,0),
    (-1,2): (2,0),
    (0,2): (2,0),
    (1,2): (-2,0),
    (2,2): (-2,0),
    (-2,1): (2,0),
    (-1,1): (2,0),
    (0,1): (2,0),
    (1,1): (-2,0),
    (2,1): (-2,0),
    (-2,0): (2,2),
    (-1,0): (2,2),
    (0,0): (2,2),
    (1,0): (-2,2),
    (2,0): (-2,2)
}

# prompts
prompt_15shot = """
You are an expert in human emotions. Below is a list of short texts, where each text describes how a person feels today. Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed. 
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Examples:
1. I have been feeling somewhat down . I am trying to be more productive and get things done . I am having trouble finding the energy and motivation . I can ' t type any more characters . I ' m pretty sure this is over 200 . → "Somewhat sluggish"
2. Calm , Content , Happy , Relaxed → "Very happy"
3. I feel okay today , but of course I’m off today . I wanted a drink last night , to be honest , but I never drank one . Yesterday was a long , bad day . I just went to bed instead , and got a good nights rest . I feel so much better today . → "Neutral"
4. trapped , stuck , anxious , confused , lost . → "Very sad"
5. I am all nervous .   I just took a test , but I won ' t transcribe and turn it in because I know it ' s not a pass .   I ' m all worried about the election and scared of checking election results .   They were bad the last time I looked , even though I know we won ' t know for a few days yet . → "Jittery, nervous"
6. A little nauseous but energetic . I cleaned for 3-4 hours and got a good workout from that which helped . Overall , I’m feeling ok besides nausea which is a common occurrence for me . The cleaning chemicals may have also contributed ? → "Active" 
7. calm , content , relaxed , tired , still → "Somewhat content"
8. I am feeling drained and irritable . I am still very exhausted and I am bored because I am too tired to do anything . I am irritable perhaps because I am bored , or perhaps because I am still so tired . I am still having a lot of brain fog and I feel slow . → "Somewhat sad"
9. energetic , happy , smiling , excited , ready → "Somewhat lively"
10. Chill , Relaxed , Calm , Mellow , Grateful → "Content, calm"
11. I am tired and having anxiety feeling very anxious . I am off today so that is helpful . I am a little jittery feel unable to relax . My head feels empty which is causing more anxiety . I will most likely drink today . → "Somewhat jittery"
12. Lively , Active , Relaxed , Charming , Energetic → "Lively, enthusiastic"
13. I am happy to be here and looking forward to the week . I will be going on a short trip with my family and looking to have a good time on it . Schools will be opening soon and the kids are happy to gi back . I hope they have a good school year . → "Somewhat happy"
14. Tired , Dehydrated , Sluggish , Sick , Numb → "Sluggish, tired"
15. Today it’s raining and rain gets me in a sleep and calm mood . I just want to lay down all day , relax and watch movies while having a cool pumpkin beer . I feel good today . And after drinking my beers I will feel more relaxed . → "Quiet"
Output format example: 
{{text_id1: “emotion1”, text_id2: “emotion2”, text_id3: “emotion3”}} 
List of texts:
{}
"""
prompt_10shot = """
You are an expert in human emotions. Below is a list of short texts, where each text describes how a person feels today. Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed. 
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Examples:
1. I have been feeling somewhat down . I am trying to be more productive and get things done . I am having trouble finding the energy and motivation . I can ' t type any more characters . I ' m pretty sure this is over 200 . → "Somewhat sluggish"
2. Calm , Content , Happy , Relaxed → "Very happy"
3. I feel okay today , but of course I’m off today . I wanted a drink last night , to be honest , but I never drank one . Yesterday was a long , bad day . I just went to bed instead , and got a good nights rest . I feel so much better today . → "Neutral"
4. trapped , stuck , anxious , confused , lost . → "Very sad"
5. I am all nervous .   I just took a test , but I won ' t transcribe and turn it in because I know it ' s not a pass .   I ' m all worried about the election and scared of checking election results .   They were bad the last time I looked , even though I know we won ' t know for a few days yet . → "Jittery, nervous"
6. A little nauseous but energetic . I cleaned for 3-4 hours and got a good workout from that which helped . Overall , I’m feeling ok besides nausea which is a common occurrence for me . The cleaning chemicals may have also contributed ? → "Active" 
7. calm , content , relaxed , tired , still → "Somewhat content"
8. I am feeling drained and irritable . I am still very exhausted and I am bored because I am too tired to do anything . I am irritable perhaps because I am bored , or perhaps because I am still so tired . I am still having a lot of brain fog and I feel slow . → "Somewhat sad"
9. energetic , happy , smiling , excited , ready → "Somewhat lively"
10. Chill , Relaxed , Calm , Mellow , Grateful → "Content, calm"
Output format example: 
{{text_id1: “emotion1”, text_id2: “emotion2”, text_id3: “emotion3”}} 
List of texts:
{}
"""
prompt_6shot = """
You are an expert in human emotions. Below is a list of short texts, where each text describes how a person feels today. Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed. 
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Examples:
1. I have been feeling somewhat down . I am trying to be more productive and get things done . I am having trouble finding the energy and motivation . I can ' t type any more characters . I ' m pretty sure this is over 200 . → "Somewhat sluggish"
2. Calm , Content , Happy , Relaxed → "Very happy"
3. I feel okay today , but of course I’m off today . I wanted a drink last night , to be honest , but I never drank one . Yesterday was a long , bad day . I just went to bed instead , and got a good nights rest . I feel so much better today . → "Neutral"
4. trapped , stuck , anxious , confused , lost . → "Very sad"
5. I am all nervous .   I just took a test , but I won ' t transcribe and turn it in because I know it ' s not a pass .   I ' m all worried about the election and scared of checking election results .   They were bad the last time I looked , even though I know we won ' t know for a few days yet . → "Jittery, nervous"
6. A little nauseous but energetic . I cleaned for 3-4 hours and got a good workout from that which helped . Overall , I’m feeling ok besides nausea which is a common occurrence for me . The cleaning chemicals may have also contributed ? → "Active" 
Output format example: 
{{text_id1: “emotion1”, text_id2: “emotion2”, text_id3: “emotion3”}} 
List of texts:
{}
"""
prompt_0shot = """
You are an expert in human emotions. Below is a list of short texts, where each text describes how a person feels today. Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed. 
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Output format example: 
{{text_id1: “emotion1”, text_id2: “emotion2”, text_id3: “emotion3”}} 
List of texts:
{}
"""

prompt_15shot_essays = """
You are an expert in human emotions. Below is a list of short texts, where each text describes how a person feels today. Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed. 
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Examples:
1. I am all nervous .   I just took a test , but I won ' t transcribe and turn it in because I know it ' s not a pass .   I ' m all worried about the election and scared of checking election results .   They were bad the last time I looked , even though I know we won ' t know for a few days yet . → "Jittery, nervous"
2. I am feeling better than yesterday and have a better mood as well . I am still low on my self esteem and want to love myself . I hope to be a parent and partner for my family . I want to enjoy my weekend with them and cook delicious meals . → "Very sad"
3. I feel super sluggish because I might have eaten something that did not agree with my stomach so I did not sleep well at all and kept getting sick through the night . I feel like I did not drink too much so it had to be the food that I ate . I wish I did not have to work today because I know I’ll feel sluggish all day . → "Sluggish, tired"
4. I have been feeling somewhat down . I am trying to be more productive and get things done . I am having trouble finding the energy and motivation . I can ' t type any more characters . I ' m pretty sure this is over 200 . → "Somewhat sluggish"
5. I am feeling drained and irritable . I am still very exhausted and I am bored because I am too tired to do anything . I am irritable perhaps because I am bored , or perhaps because I am still so tired . I am still having a lot of brain fog and I feel slow . → "Somewhat sad"
6. I am tired and having anxiety feeling very anxious . I am off today so that is helpful . I am a little jittery feel unable to relax . My head feels empty which is causing more anxiety . I will most likely drink today . → "Somewhat jittery"
7. A little nauseous but energetic . I cleaned for 3-4 hours and got a good workout from that which helped . Overall , I’m feeling ok besides nausea which is a common occurrence for me . The cleaning chemicals may have also contributed ? → "Active"
8. I feel okay today , but of course I’m off today . I wanted a drink last night , to be honest , but I never drank one . Yesterday was a long , bad day . I just went to bed instead , and got a good nights rest . I feel so much better today → "Neutral"
9. Today it’s raining and rain gets me in a sleep and calm mood . I just want to lay down all day , relax and watch movies while having a cool pumpkin beer . I feel good today . And after drinking my beers I will feel more relaxed . → "Quiet"
10. I am currently feeling a little tired but calm and content . I am not overly joyous , but I do not feel sad or anxious at the moment. I am focused on my work for the day but I do not feel stressed about it . → "Somewhat content"
11. I am happy to be here and looking forward to the week . I will be going on a short trip with my family and looking to have a good time on it . Schools will be opening soon and the kids are happy to gi back . I hope they have a good school year . → "Somewhat happy"
12. I am feeling good right now it ' s a semi nice day out and I ' m out with my friend and her dog going on a nice walk . We are enjoying the fresh fall air for a few hours then we will go back home and do whatever house work we have to do → "Somewhat lively"
13. Just finished my clinical rotation for today and walking towards my husband’s office . Although the walk will be at least an hour and a half , I am still very excited and I think this is a very good exercise for me . → "Lively, enthusiastic"
14. I’m feeling very relaxed and about to get ready for work at four today .   It’s taco Tuesday and should be busy , also ORG is surging which makes me happy as well .   I tried to nap before work but couldn’t due to my roommate coming home early from work . → "Very happy"
15. I went to the gym after work and made dinner at home so I feel relaxed and accomplished . I feel calmer than I felt in the morning when I woke up because I was stressed out in the morning about the things I have to accomplish all through the day . → "Content, calm"
Output format example: 
{{text_id1: “emotion1”, text_id2: “emotion2”, text_id3: “emotion3”}} 
List of texts:
{}
"""
prompt_15shot_words = """
You are an expert in human emotions. Below is a list of short texts, where each text describes how a person feels today. Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed. 
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Examples:
1. Frantic , Anxious , Panicked , Stressed , Overwhelmed → "Jittery, nervous"
2. trapped , stuck , anxious , confused , lost → "Very sad"
3. Tired , Dehydrated , Sluggish , Sick , Numb → "Sluggish, tired"
4. Tired , Not motivated , Lazy , Happy , Sad → "Somewhat sluggish"
5.  Tired , Sleepy , Annoyed , Frustrated , Bored → "Somewhat sad"
6. Anxious , Relieved , Tired , Wired , Anticipating → "Somewhat jittery"
7.  Active , Excited , Hoped , Enthusiastic , Brave → "Active"
8. Content , Calm , Neutral , Pleasant , Happy → "Neutral"
9.  Focused , Determined , Joyful , Relaxed , Good → "Quiet"
10. calm , content , relaxed , tired , still → "Somewhat content"
11. Healthy , Calm , Happy , Content , Excited → "Somewhat happy"
12. energetic , happy , smiling , excited , ready → "Somewhat lively"
13. Lively , Active , Relaxed , Charming , Energetic → "Lively, enthusiastic"
14. Calm , Content , Happy , Relaxed → "Very happy"
15. Chill , Relaxed , Calm , Mellow , Grateful → "Content, calm"
Output format example: 
{{text_id1: “emotion1”, text_id2: “emotion2”, text_id3: “emotion3”}} 
List of texts:
{}
"""
prompt_30shot_words = """
You are an expert in human emotions. Below is a list of short texts, where each text describes how a person feels today. Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed. 
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Examples:
1. Frantic , Anxious , Panicked , Stressed , Overwhelmed → "Jittery, nervous"
2. trapped , stuck , anxious , confused , lost → "Very sad"
3. Tired , Dehydrated , Sluggish , Sick , Numb → "Sluggish, tired"
4. Tired , Not motivated , Lazy , Happy , Sad → "Somewhat sluggish"
5. Tired , Sleepy , Annoyed , Frustrated , Bored → "Somewhat sad"
6. Anxious , Relieved , Tired , Wired , Anticipating → "Somewhat jittery"
7. Active , Excited , Hoped , Enthusiastic , Brave → "Active"
8. Content , Calm , Neutral , Pleasant , Happy → "Neutral"
9. Focused , Determined , Joyful , Relaxed , Good → "Quiet"
10. calm , content , relaxed , tired , still → "Somewhat content"
11. Healthy , Calm , Happy , Content , Excited → "Somewhat happy"
12. energetic , happy , smiling , excited , ready → "Somewhat lively"
13. Lively , Active , Relaxed , Charming , Energetic → "Lively, enthusiastic"
14. Calm , Content , Happy , Relaxed → "Very happy"
15. Chill , Relaxed , Calm , Mellow , Grateful → "Content, calm"
16. Stressed , Nervous , Tired , Sleepy , Unfocused → "Jittery, nervous"
17.  Sad , Depressed , Worried , Alone , Afraid → "Very sad"
18. Tired , Sluggish , Disappointed , Dazed , Weird → "Sluggish, tired"
19. Sluggish , Uncomfortable , Lethargic , Anxious , Nervous → "Somewhat sluggish"
20.  Nauseous , Lethargic , Afraid , Overwhelmed , Sensitive → "Somewhat sad"
21.  Overstimulated , Overwhelmed , Exhausted , Irritable , Fragile → "Somewhat jittery"
22.  Hungry , Energized , Curious , Determined , Confident→ "Active"
23. Overwhelming , Tired , Busy , Hungry , Bored → "Neutral"
24.  Tired , Exhausted , Calm , Candid , Humored → "Quiet"
25.  Sore , Sleepy , Warm , Content , Relaxed→ "Somewhat content"
26.  Relieved , Content , Calm , Loved , Lucky → "Somewhat happy"
27. Awake , Alive , Energized , Happy , Alive → "Somewhat lively"
28. happy , content , full , tired , relaxed → "Lively, enthusiastic"
29. Excited , Happy , Good , Smiling , Relaxed → "Very happy"
30. Relaxed , Content , Hungry , Calm , Tired → "Content, calm"
Output format example: 
{{text_id1: “emotion1”, text_id2: “emotion2”, text_id3: “emotion3”}} 
List of texts:
{}
"""

prompt_user_aware_static_emotion = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed.
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples how this specific user tends to describe their emotional states, and apply this understanding when labeling the new texts. Your labels should follow the user’s own pattern of emotional expression, not generic interpretations. 
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- All texts in the evaluation set come from the same user as the examples.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Allowed emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Previous texts with assigned emotions:
{train}
Sequence of texts for evaluation:
{predict}
"""

# EXTRA_EXPERIMENT
# Before assigning values, internally:
# - infer user-specific valence prototypes from the labeled examples,
# - compare each new text to the most similar prototypes,
# (INCLUDED) - prefer the weakest valence value that sufficiently fits the text unless there is clear evidence for stronger intensity.

prompt_user_aware_static_valence = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign a single valence value to each text, using the scale from -2 to +2 that best matches the overall emotional valence expressed.
Valence scale:
-2 = clearly negative
-1 = moderately negative
 0 = neutral or mixed
+1 = moderately positive
+2 = clearly positive
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples how this specific user tends to describe their emotional states, and apply this understanding when labeling the new texts. Your labels should follow the user’s own pattern of emotional expression, not generic interpretations.
Instructions:
- Prefer the weakest valence value that sufficiently fits the text unless there is clear evidence for stronger intensity.
- If the text does not explicitly describe feelings, choose the valence value that best fits the emotional state implied by the text.
- All texts in the evaluation set come from the same user as the examples.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments.
Previous texts with assigned valence:
{train}
Sequence of texts for evaluation:
{predict}
"""

prompt_user_aware_static_arousal = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign a single arousal value to each text, using the scale from 0 to 2 that best matches the overall level of emotional activation expressed.
Arousal reflects how activated the emotional state is, regardless of whether it is positive or negative.
Arousal scale:
0 = low activation (calm, flat, drained, quiet)
1 = moderate activation (alert, engaged, uneasy)
2 = high activation (overwhelmed, restless, excited, agitated)
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples how this specific user tends to describe their emotional states, and apply this understanding when labeling the new texts. Your labels should follow the user’s own pattern of emotional expression, not generic interpretations.
Instructions:
- Prefer the weakest arousal value that sufficiently fits the text unless there is clear evidence for stronger activation.
- Do not infer higher arousal solely from longer text or multiple emotion words.
- If the text does not explicitly describe feelings, choose the arousal value that best fits the level of activation implied by the text.
- All texts in the evaluation set come from the same user as the examples.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments.
Previous texts with assigned arousal:
{train}
Sequence of texts for evaluation:
{predict}
"""

prompt_user_aware_static_val_and_aro = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign TWO numerical values to each text:
- a single valence value (emotional positivity vs negativity)
- a single arousal value (emotional activation level)
Valence and arousal must be inferred independently.
Valence reflects how positive or negative the emotional state is.
Valence scale:
-2 = clearly negative
-1 = moderately negative
 0 = neutral or mixed
+1 = moderately positive
+2 = clearly positive
Arousal reflects how activated the emotional state is, regardless of whether it is positive or negative.
Arousal scale:
0 = low activation (calm, flat, drained, quiet)
1 = moderate activation (alert, engaged, uneasy)
2 = high activation (overwhelmed, restless, excited, agitated)
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples how this specific user tends to describe their emotional states, and apply this understanding when labeling the new texts. Your labels should follow the user’s own pattern of emotional expression, not generic interpretations.
INSTRUCTIONS
- Prefer the weakest value that sufficiently fits the text unless there is clear evidence for stronger intensity or activation.
- Valence and arousal are independent: positive or negative texts may have low or high arousal.
- If the text does not explicitly describe feelings, infer valence and arousal from the implied emotional state.
- All texts in the evaluation set come from the same user as the examples.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments.
Output format example (JSON-like):
{{{{text_id1: {{valence: 1, arousal: 0}}, text_id2: {{valence: -1, arousal: 1}}, text_id3: {{valence: 2, arousal: 0}}}}
Previous texts with assigned valence and arousal:
{train}
Sequence of texts for evaluation:
{predict}
"""

prompt_dynamic = """
You are an expert in human emotions. Below is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed.
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples how this specific user tends to describe their emotional states, and apply this understanding when labeling the new texts. Your labels should follow the user’s own pattern of emotional expression, not generic interpretations. 
Important clarification:
The texts to be labeled are a direct chronological continuation of the previously labeled history.
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- All texts in the evaluation set come from the same user as the examples.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Allowed emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Previously labeled history (earliest to latest):
{train}
labeled continuation (chronologically follows the history above):
{predict}
"""

fake_words_history = ["Calm , Indifferent , Present , Mindful , Chill → Neutral",
"Happy , Comforted , Pampered , Loved , Joyful → Somewhat happy",
"Tired , Sore , Sleepy , Heavy , Groggy → Sluggish, tired",
"Aware , Antsy , Impatient , Gassy , Full → Somewhat jittery",
"Peaceful , Calm , Warm , Fluid , Curious → Somewhat sluggish",
"Warm , Tired , Unmotivated , Puddle , Full → Somewhat sluggish",
"Calm , Refreshed , Introspective , Enriched , Meager → Somewhat happy",
"Chill , Calm , Satisfied , Warm , Leisurely → Content, calm",
"Energized , Active , Mobile , Motivated , Happy → Active",
"Motivated , Proud , Settled , Talented , Fitted → Neutral"]

fake_essays_history = [
"I have just been hanging out at home for a few days so I just feel calm and content . I have to go to work today but I feel pretty good about it just because I haven't seen anyone in a few days so it will be nice . → Content, calm",
"I am feeling pretty calm and content . I had a easy shift at work and it was nice to see everyone even though it was a little slow . It was weird to have to talk to so many people after I was at home by myself for awhile . → Somewhat content",
"It's been a pretty good day so I am just happy and relaxed . I got most of my house cleaned so that makes me feel more calm when I have my house clean . I'm gonna meet some friends later so I'm happy about that . → Somewhat happy",
"I stayed out late last night because I was hanging out with friends and having a good time . I woke up early and could not fall back asleep so I am just tired . I don't have a whole lot going on today so I am excited I can lounge around for a while . → Somewhat sluggish",
"I'm feeling okay now . It's a pretty day so that makes me feel happy . I haven't done a whole lot today so I am just hanging out and doing some homework . It is nice not to have to rush to get things done since I got two days off early in the week . → Neutral",
"I feel calm and just sleepy . I had a good day just hanging out so I am just ready for bed and get ready for tomorrow . Today was pretty uneventful just a quiet day . → Quiet",
"I have a exciting day today so I am pretty happy . I didn't sleep too great last night so I just feel a little sluggish . I have a lot to do today but I get to go to a pep rally with my friends tonight so I am pretty excited about that . → Content, calm.",
"I got to leave work early and hang out with my friends so I am super happy . Our hometown team is going to super bowl so we went to the pep rally and even though it was freezing it was so awesome so I am feeling very happy . → Very happy",
"I went out to celebrate with a few of my friends last night and had a fun time I did not really drink because we did not stay out long . I got home early but did not sleep well so I just feel super sluggish and sleepy today . → Somewhat sluggish",
"I have been having a rough day and I just feel tired and very sluggish . I had a long day at school and it is busy at work today as well so I just feel like I am dragging my feet all day . → Sluggish, tired",
"I am just feeling very neutral and relaxed . I had a long day at work and school and I just want to sit down and hang out at the house . I just feel exhausted because I did not get good sleep last night and wish I could have slept better . → Neutral"
]

prompt_user_aware_static_openai_instructions = """
You are an expert in human emotions. You are provided with is a chronological sequence of short texts written by the same user, each describing how they felt on a particular day.
Your task is to assign exactly one emotion from the allowed list to each text — the emotion that best matches the overall feeling expressed.
The user has a personal, consistent way of expressing emotions. Learn from the previously labeled examples how this specific user tends to describe their emotional states, and apply this understanding when labeling the new texts. Your labels should follow the user’s own pattern of emotional expression, not generic interpretations. 
Instructions:
- If the text does not explicitly describe feelings, choose the emotion that best fits the emotional state implied by the text.
- All texts in the evaluation set come from the same user as the examples.
- Preserve the original order of the texts.
- Each text must appear exactly once in the output — no duplicates and none omitted.
- Present the result in a plain Python-friendly dictionary format, without any explanations or comments. 
Allowed emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Previous texts with assigned emotions:
{train}
"""

prompt_user_aware_static_openai_input = """"
Assign exactly one emotion from the allowed list to each text.
Sequence of texts for evaluation:
{predict}
"""

prompt_state_change = """
You are an expert in human emotion dynamics and affective patterns. 
You are provided with a chronological sequence of short texts written by the same user. Each text describes how the user felt on a particular day and is paired with then assigned emotion label.
Your task is to predict the single most likely emotion this user will experience on the next day following the provided sequence.
Important constraints:
- The user has a stable, personal way of expressing emotions across days.
- Infer temporal patterns, trends, and transitions in the user’s emotional state.
- Base your prediction on how this specific user’s emotions typically evolve over time, not on generic population averages.
Output requirements:
- Select exactly one emotion from the allowed list below.
- Output only the emotion label, without any explanations or comments. 
Allowed emotions:
{{"Jittery, nervous", "Somewhat jittery", "Active", "Somewhat lively", "Lively, enthusiastic", "Very sad", "Somewhat sad", "Neutral", "Somewhat happy", "Very happy", "Sluggish, tired", "Somewhat sluggish", "Quiet", "Somewhat content", "Content, calm"}} 
Chronological history (earliest to latest):
{train}
"""

prompt_state_change_valence = """
You are an expert in human emotion dynamics and affective patterns. 
You are provided with a chronological sequence of short texts written by the same user. Each text describes how the user felt on a particular day and is paired with then assigned emotion label.
Your task is to predict the single most likely valence value this user will experience on the next day following the provided sequence.
Valence reflects how positive or negative the emotional state is.
Valence scale:
-2 = very negative (distressed, hopeless, miserable)
-1 = somewhat negative (down, dissatisfied, uneasy)
 0 = neutral or mixed (flat, indifferent, balanced)
 1 = somewhat positive (pleasant, mildly happy, content)
 2 = very positive (happy, joyful, enthusiastic)
Important constraints:
- The user has a stable, personal way of expressing emotions across days.
- Infer temporal patterns, trends, and transitions in the user’s emotional state.
- Base your prediction on how this specific user’s emotions typically evolve over time, not on generic population averages.
Output requirements:
- Output exactly one numerical valence value from the set {{-2, -1, 0, 1, 2}}.
- Output only the value, without any explanations or comments. 
Chronological history (earliest to latest):
{train}
"""


prompt_state_change_arousal = """
You are an expert in human emotion dynamics and affective patterns. 
You are provided with a chronological sequence of short texts written by the same user. Each text describes how the user felt on a particular day and is paired with then assigned emotion label.
Your task is to predict the single most likely arousal value this user will experience on the next day following the provided sequence.
Arousal reflects how activated the emotional state is, regardless of whether it is positive or negative.
Valence scale:
Arousal scale:
0 = low activation (calm, flat, drained, quiet)
1 = moderate activation (alert, engaged, uneasy)
2 = high activation (overwhelmed, restless, excited, agitated)
Important constraints:
- The user has a stable, personal way of expressing emotions across days.
- Learn from the previously labeled examples how this specific user expresses low, moderate, and high activation.
- Infer temporal patterns, trends, and transitions in arousal rather than treating days independently.
- Base your prediction on how this specific user’s emotions typically evolve over time, not on generic population averages.
Output requirements:
- Output exactly one numerical arousal value from the set {{0, 1, 2}}.
- Output only the value, without any explanations or comments. 
Chronological history (earliest to latest):
{train}
"""

# EXAMPLE VALENCE
# Tired , Stressed , Exhausted , Annoyed -> -1
# Exhausted , Invisible , Sore , Morose  -> 0
# Tired , Unappreciated , Lonely  -> 1
# Frustrated , Tired , Annoyed  -> 0
# Annoyed , Frustrated , Unheard , Misunderstood  -> -1
# Tired , Frustrated , Misunderstood  -> -2
# Invisible , Drained , Frustrated  -> -2
# Drained , Sluggish , Annoyed  -> 0
# Refreshed , Energetic , Enthusiastic , Hopeful  -> 2
# Blah , Quiet , Uninterested , Ignored  -> 0
# Refreshed , Releived , Content  -> 1
# Annoyed , Frustrated , Anxious  -> 0
# Neutral , Blah , Uninspired  -> 0
# Overwhelmed , Lonely , Annoyed  -> -2
# Excited , Anxious , Calm , Mellow  -> 2