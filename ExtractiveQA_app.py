import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors

## Input : KEY, pdf path , question
## output : answer

class Data:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path 
        self.start_page = 1
        self.end_page = None
        self.word_length = 150

    def _preprocess(self,text):
        '''
        preprocess chunks
        1. Replace new line character with whitespace.
        2. Replace redundant whitespace with a single whitespace
        '''
        text = text.replace('\n', ' ')
        text = re.sub('\s+', ' ', text)
        text = re.sub(r'\\u[e-f][0-9a-z]{3}',' ', text)
        return text
    
    def _pdf_to_text(self):
        '''
            convert pdf to a list of words.
        '''
        doc = fitz.open(self.pdf_path)
        total_pages= doc.page_count

        if self.end_page is None:
            self.end_page = total_pages
        text_list=[]

        for i in tqdm(range(self.start_page-1, self.end_page)):
            text= doc.load_page(i).get_text('text')
            text= self._preprocess(text)
            text_list.append(text)
        doc.close()
        return text_list
    
    def text_to_chunk(self):
        ''''
            converts the text into smaller chunks of word_length
        '''
        word_length= self.word_length
        texts = self._pdf_to_text()
        tokens = [text.split(' ') for text in texts]
        chunks=[]
        for idx, words in enumerate(tokens):
            for i in range(0,len(words), word_length):
                chunk = words[i:i+word_length]
                if (i+word_length) > len(words) and (len(chunk) < word_length) and (len(tokens) != (idx+1)):
                    tokens[idx+1] = chunk + tokens[idx+1]
                    continue
                chunk = ' '.join(chunk).strip()
                chunk=  f'[{idx+self.start_page}]'+' '+'"'+chunk+'"'
                chunks.append(chunk)
        return chunks


class SemanticSearch:
    def __init__(self):
        ## The encoder that will embed the texts.
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False


    def _get_text_embedding(self, texts, batch=1000):
        '''
            Returns embeddings for the tokens using the 
            universal sentence encoder from tensorflow hub.
        '''
        embeddings = []
        for i in tqdm(range(0, len(texts), batch)):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

    def fit(self, data, batch=1000, n_neighbors=5):
        '''
            The chunks will be fit so that the chunks with most semantic 
            similarity with the question asked will be returned in sorted manner
        '''
        self.data = data
        self.embeddings = self._get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

class Model:
    def __init__(self,KEY,pdf_path, question):
        self.KEY = KEY
        self.data = Data(pdf_path)
        self.recommender = SemanticSearch()
        self.question = question

    def _fetch_ordered_chunks(self):
        chunks = self.data.text_to_chunk()
        self.recommender.fit(chunks)
        ordered_chunks = self.recommender(self.question)
        return ordered_chunks
        
    def _createQuestionPrompt(self, question, n):
        topn_chunks = self._fetch_ordered_chunks()
        prompt= ""
        prompt += 'search results:\n\n'
        for c in topn_chunks:
            prompt+=c+'\n\n'
        prompt += "Instructions: Compose a comprehensive reply to the query using the search results given."\
              "Cite each reference using [number] notation (every result has this number at the beginning)."\
              "Citation should be done at the end of each sentence. If the search results mention multiple subjects"\
              "with the same name, create separate answers for each. Only include information found in the results and"\
              "don't add any additional information. Make sure the answer is correct and don't output false content."\
              "If the text does not relate to the query, simply state 'Found Nothing'. Don't write 'Answer:'"\
              "Directly start the answer.\n"
        prompt+= f"Query : {question} \n\n"
        return prompt
    
    def generateAnswer(self,n=3,engine ='text-davinci-003' ):
        prompt= self._createQuestionPrompt(self.question, n)
        openai.api_key = self.KEY
        completions = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=256,
            n=1,
            temperature=0.9,
        )
        answer = completions.choices[0]['text'].replace('.','\n\n')
        return answer