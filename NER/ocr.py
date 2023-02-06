import io
from google.cloud import vision
import os
from tqdm import tqdm
#
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\AkhileshSharma\Desktop\data\kn-key-extractors\gcp\googlecreds.json'


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

# vertexList = detectText('1357441524-66751914-CI6456DreammakerPO119393-1.png')[0]
# print(vertexList[0])
# for vertexText in vertexList:
#     info = {}
#     text = vertexText[0]
#     vertex = vertexText[1]
#     x1 = vertex[0].x
#     y1 = vertex[0].y
#     x3 = vertex[2].x
#     y3 = vertex[2].y
#     print(x1,y1,x3,y3)


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



# for i, j in tqdm(enumerate(os.listdir('n'))):
#     quality = detect_document(os.path.join('n', j))
#     os.rename(os.path.join('n', j), os.path.join('2', os.path.splitext(j)[0]+'_'+str(quality)[:2]+'.png'))

# print(detect_document('1200311944-PCQikTlkNgmYaUUf2Wfb-FC-XXX_STPC_12Jan23-2105-3.png'))