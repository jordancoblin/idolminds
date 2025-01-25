import argparse
import subprocess
import sys
import importlib
from RealtimeTTS import TextToAudioStream, EdgeEngine, EdgeVoice

def check_and_install_packages(packages):
    """
    Checks if the specified packages are installed, and if not, prompts the user
    to install them.

    Parameters:
    - packages: A list of dictionaries, each containing:
        - 'module_name': The module or package name to import.
        - 'attribute': (Optional) The attribute or class to check within the module.
        - 'install_name': The name used in the pip install command.
        - 'version': (Optional) Version constraint for the package.
    """
    for package in packages:
        module_name = package['module_name']
        attribute = package.get('attribute')
        install_name = package.get('install_name', module_name)
        version = package.get('version', '')

        try:
            # Attempt to import the module
            module = importlib.import_module(module_name)
            # If an attribute is specified, check if it exists
            if attribute:
                getattr(module, attribute)
        except (ImportError, AttributeError):
            user_input = input(
                f"This program requires '{module_name}'"
                f"{'' if not attribute else ' with attribute ' + attribute}, which is not installed or missing.\n"
                f"Do you want to install '{install_name}' now? (y/n): "
            )
            if user_input.strip().lower() == 'y':
                try:
                    # Build the pip install command
                    install_command = [sys.executable, "-m", "pip", "install"]
                    if version:
                        install_command.append(f"{install_name}{version}")
                    else:
                        install_command.append(install_name)

                    subprocess.check_call(install_command)
                    # Try to import again after installation
                    module = importlib.import_module(module_name)
                    if attribute:
                        getattr(module, attribute)
                    print(f"Successfully installed '{install_name}'.")
                except Exception as e:
                    print(f"An error occurred while installing '{install_name}': {e}")
                    sys.exit(1)
            else:
                print(f"The program requires '{install_name}' to run. Exiting...")
                sys.exit(1)

if __name__ == "__main__":
    print_voices = False
    print_detailed_voices = False

    # from install_packages import check_and_install_packages
    import time
    import signal

    check_and_install_packages([
        {
            'module_name': 'keyboard',                    # Import module
            'install_name': 'keyboard',                   # Package name for pip install
        },
        {
            'module_name': 'RealtimeTTS',                # Import module
            'install_name': 'RealtimeTTS',               # Package name for pip install
            'attribute': 'EdgeEngine'                     # Check for EdgeEngine class
        }
    ])


    import keyboard

    parser = argparse.ArgumentParser(description="Realtime TTS script with pause/resume functionality.")
    parser.add_argument("-l", "-voices", "-list", "--list-voices", "--listvoices", nargs='*', default=None, 
                        help="List voices. Optional filters: language code and/or gender (e.g. 'en', 'male', 'en female')")
    parser.add_argument("-dl", "-detailedvoices", "--detailed-voices", nargs='*', default=None,
                        help="List detailed voice info. Optional filters: language code and/or gender")
    parser.add_argument("-t", "-text", "--text", type=str, default="", help="Text to convert to speech")
    parser.add_argument("-v", "--voice", type=str, default="en-US-JennyNeural", help="Voice to use for speech synthesis")
    parser.add_argument("-p", "--pitch", type=int, default=0,
                        help="Speech pitch adjustment (-50 to +50)")
    parser.add_argument("-r", "-s", "--rate", "--speed",  type=int, default=0,
                        help="Speech rate adjustment (-50 to +50)")
    parser.add_argument("-vol", "--volume", type=int, default=0,
                        help="Speech volume adjustment (-50 to +50)")
    args = parser.parse_args()

    for param in ['pitch', 'rate', 'volume']:
        value = getattr(args, param)
        if not -50 <= value <= 50:
            parser.error(f"{param} must be between -50 and +50")

    text = args.text
    voice = args.voice

    if args.list_voices:
        print_voices = True
    
    if args.detailed_voices:
        print_detailed_voices = True

    def dummy_generator():
        if text:
            yield text
            return
        yield "This is a longer test with multiple sentences to demonstrate pause and resume functionality. "
        yield "Press spacebar to pause or resume playback. Press Q to quit. "
        yield "Quick technical note: When you press pause, there may be a brief delay before the audio stops with EdgeEngine. "
        yield "This happens because the audio player (MPV) pre-buffers some audio data for smooth playback - like water still flowing from a pipe even after you close the tap. "

    print(EdgeEngine)

    engine = EdgeEngine(
        rate=args.rate,
        pitch=args.pitch,
        volume=args.volume
    )


    # Modified language filtering section
    if args.list_voices is not None or args.detailed_voices is not None:
        filters = args.list_voices if args.list_voices is not None else args.detailed_voices
        filters = [f.lower() for f in filters] if filters else []
        
        # Extract gender filter first
        gender_filter = next((f for f in filters if f in ['male', 'female']), '')
        
        # Everything else is treated as a language filter
        lang_filters = [f for f in filters if f != gender_filter]
        
        all_voices = engine.get_voices()
        print("Available voices:")
        displayed_count = 0
        
        for i, voice in enumerate(all_voices):
            # Skip if gender filter doesn't match
            if gender_filter and voice.gender.lower() != gender_filter:
                continue
                
            # Check if any language filter matches
            if lang_filters:
                matches_lang = False
                for lang_filter in lang_filters:
                    if len(lang_filter) == 2:
                        # For 2-char codes, match only the language part
                        matches_lang = voice.locale.lower().startswith(lang_filter.lower())
                    else:
                        # For longer codes, match the full locale
                        matches_lang = lang_filter.lower() in voice.locale.lower()
                    if matches_lang:
                        break
                if not matches_lang:
                    continue
                
            displayed_count += 1
            if args.detailed_voices is not None:
                print(f"Voice {displayed_count}:\n{repr(voice)}")
            else:
                print(f"{displayed_count}. {voice}")
        
        if displayed_count == 0:
            print("No voices found matching the specified filters.")
        exit(0)

    stream = TextToAudioStream(engine)
    if voice:
        engine.set_voice(voice)

    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Cleaning up...")
        stream.stop()
        import sys
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("Starting stream - Press SPACE to pause/resume, Q to quit")
    stream.feed(dummy_generator()).play_async(log_synthesized_text=True)

    is_playing = True
    while True:
        if not stream.is_playing():
            print("\nStream finished.")
            break

        if keyboard.is_pressed('space'):
            if is_playing:
                stream.pause()
                print("Paused")
            else:
                stream.resume()
                print("Resumed")
            is_playing = not is_playing
            time.sleep(0.1)  # Debounce
            
        if keyboard.is_pressed('q'):
            print("\nQuitting...")
            break
            
        time.sleep(0.1)
    
    stream.stop()