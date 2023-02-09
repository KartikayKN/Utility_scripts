import io
# import cv2
from google.cloud import vision
import os
# from tqdm import tqdm
# import re
# from pascal_voc_writer import Writer
# from googletrans import Translator
# translator = Translator(raise_exception=True)
# import xml.etree.ElementTree as ET
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join('.', 'gcp/googlecreds.json')


def detectText(path):

    vertexList = []
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    # print(response)
    texts = response.text_annotations
    bool = False
    complete_text = ''
    for text in texts:
        # print(text)

        vertices = text.bounding_poly.vertices
        if bool:
            vertexList.append((text.description, vertices))
        else:
            complete_text = text.description
            bool = True

    if response.error.message:
        raise Exception(
            '{}\n'.format(response.error.message))

    return vertexList, complete_text


def detect_document(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_document_text_detection]
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        goodWords = 0
        count = 0
        for block in page.blocks:
            # print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                # print('Paragraph confidence: {}'.format(
                #     paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    count += 1
                    if float(word.confidence) > 0.95:
                        goodWords += 1
                    # print('Word text: {} (confidence: {})'.format(word_text, word.confidence))

                    # for symbol in word.symbols:
                    #     print('\tSymbol: {} (confidence: {})'.format(
                    #         symbol.text, symbol.confidence))

        quality = (100 * goodWords / count if count > 20 else 0)
        return quality


# path = 'final'
# import pandas as pd
# data = []
#
# for i in tqdm(os.listdir(path)):
#     if i.endswith('png'):
#         name = os.path.splitext(i)[0]
#         img = cv2.imread(os.path.join(path, i))
#         h, w = img.shape[:2]
#         vertexList, full_text = detectText(os.path.join(path, i))
#         bboxes = []
#         words = []
#         ner_tag = []
#         for vertexText in vertexList:
#             text = vertexText[0]
#             vertex = vertexText[1]
#             x1 = vertex[0].x
#             y1 = vertex[0].y
#             x3 = vertex[2].x
#             y3 = vertex[2].y
#             bboxes.append([x1, y1, x3, y3])
#             words.append(text)
        #     if x3>x1 and y3>y1:
        #         words.append(text)
        #         bboxes.append([x1, y1, x3, y3])
        #         ner_tag.append('O')
        # data.append([words, bboxes, ner_tag, './arabic_data/'+i])
        # df = pd.DataFrame(columns=['words', 'ner_tag'])
        # df['words'] = words
        # df['ner_tag'] = ner_tag
        # df.to_csv(os.path.join('resized', name+'.csv'), index=False)

# df = pd.DataFrame(data, columns=['words', 'bboxes', 'ner_tags', 'image_path'])
# df.to_csv('new_train.csv', index=False)


def detect(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_document_text_detection]
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    breaks = vision.TextAnnotation.DetectedBreak.BreakType
    paragraphs = []
    lines = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                para = ""
                line = ""
                for word in paragraph.words:
                    for symbol in word.symbols:
                        line += symbol.text
                        if symbol.property.detected_break.type_ == breaks.SPACE:
                            line += ' '
                        if symbol.property.detected_break.type_ == breaks.EOL_SURE_SPACE:
                            line += ' '
                            lines.append(line)
                            para += line
                            line = ''
                        if symbol.property.detected_break.type_ == breaks.LINE_BREAK:
                            lines.append(line)
                            para += line
                            line = ''
                paragraphs.append(para)

    print(paragraphs)
    print(lines)


# x = detectText('108005325-232431273-2-OriginalSample-3.png')[0]
#
# for i in x:
#     print(i[0])