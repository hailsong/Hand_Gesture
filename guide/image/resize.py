from PIL import Image
import os


def resize_gif(path, save_as=None, resize_to=None, ratio=None):
    all_frames = extract_and_resize_frames(path, resize_to, ratio)

    im = Image.open(path)
    framerate = get_duration_per_frames(im)

    if not save_as:
        number = 1
        if not '\\' in path:
            path = os.getcwd() + '\\' + path
        while os.path.isfile(path):
            save_as = os.path.splitext(path)[0] + str(number) + '.gif'
            if not os.path.isfile(save_as):
                break
            number += 1
        print('The resized filename will be ' + os.path.basename(save_as))

    if len(all_frames) == 1:
        print("Warning : There is only 1 frame.")
        print('The file was resized to ' + str(all_frames[0].size) + '.')
        all_frames[0].save(save_as, optimize=True)
    else:
        print('The file was resized to ' + str(all_frames[0].size) + '.')
        all_frames[0].save(save_as, optimize=True, save_all=True, append_images=all_frames[1:], loop=0,
                           duration=framerate)


def analyseImage(path):
    im = Image.open(path)
    results = {'size': im.size, 'mode': 'full'}
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    finally:
        return results


def extract_and_resize_frames(path, resize_to=None, ratio=None):
    mode = analyseImage(path)['mode']
    im = Image.open(path)
    # if not resize_to:
    #     if not ratio:
    #         ratio = 0.5
    #     resize_to = [int(ratio * s) for s in im.size]
    im = im.crop((500, 500, 500, 500))
    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')
    all_frames = []

    try:
        while True:
            if not im.getpalette():
                im.putpalette(p)
            new_frame = Image.new('RGBA', im.size)
            if mode == 'partial':
                new_frame.paste(last_frame)
            new_frame.paste(im, (0, 0), im.convert('RGBA'))
            new_frame.thumbnail(resize_to, Image.LANCZOS)
            all_frames.append(new_frame)
            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    finally:
        return all_frames


def get_duration_per_frames(image_object):
    image_object.seek(0)
    duration = 0
    try:
        while True:
            duration += image_object.info['duration']
            image_object.seek(image_object.tell() + 1)
    except EOFError:
        pass
    finally:
        duration_per_frames = round(duration / image_object.n_frames, 2)
        return duration_per_frames


resize_gif('파일명.gif', save_as=None, resize_to=None, ratio=None)