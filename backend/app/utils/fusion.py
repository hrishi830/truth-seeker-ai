def late_fusion(audio_score, video_score):

    # strong confidence cases
    if audio_score > 0.9:
        return audio_score

    if video_score > 0.9:
        return video_score

    # video more important
    return 0.2 * audio_score + 0.8 * video_score