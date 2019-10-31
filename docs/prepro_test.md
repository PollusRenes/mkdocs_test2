## Benodigde libraries

Gegevens om te verwijderen uit zienswijzen:

* Adressen
* Namen
* Bankgegevens
* E-mails

Stappen die we zetten voor preprocessing:

* remove e-mails
* remove newline characters
* remove single quotes 
* hoofdletters naar kleine letters
* getallen weg
* remove punctuatie
* remove woorden kleiner dan lengte 4
* remove 3 klinkers achter elkaar (2 voor i's)
* woorden omzetten naar lemma-vorm
* Nederlandse stopwoorden weghalen (extra toevoegen als we niets-zeggende woorden vinden)
* plaatsnamen verwijderen

Informatie over de verschillende imports.

	import pandas as pd
	import re
	import spacy
	import gensim
	from gensim.corpora.dictionary import Dictionary
	from gensim import models, corpora
	from gensim.models.coherencemodel import CoherenceModel
	from gensim.models.ldamodel import LdaModel

	nlp = spacy.load('nl_core_news_sm') # Is 34 MB in size
	dp_df = pd.read_csv('select_content_from__uploadFile__order_b.tsv', sep = '\t', names=['text'])

Informatie over het toevoegen van stopwoorden aan de default spaCy lijst.

	# Laad extra Nederlandse stopwoorden in die niet al in spacy zelf zitten
	with open('/home/datalab/Documents/DigPart/Topic_modelling/stopwords-nl.txt', 'r') as f:
	    for word in f:
		# Verwijder het \n gedeelte wat uit de .txt files mee komt
		# Voeg daarna toe aan de lijst met stopwoorden van spaCy
		if word.endswith('\n'):
		    nlp.Defaults.stop_words.add(word[:-1])

	# Doe hetzelfde, maar dan voor namen van steden
	with open('/home/datalab/Documents/DigPart/Topic_modelling/plaatsnamen_short_test.txt',
		  'r') as f2:
	    for city in f2:
		if city.endswith('\n'):
		    # Verwijder het \n gedeelte wat uit de .txt files mee komt
		    city = city[:-1]
		    city_split = city.split()
		    if len(city_split) > 1:
		        # Als namen uit meer dan één woord bestaan, neem voor nu de
		        # eerste twee woorden
		        nlp.Defaults.stop_words.add(city[0])
		        nlp.Defaults.stop_words.add(city[1])
		    else:
		        # Anders pakken we gewoon de hele naam
		        nlp.Defaults.stop_words.add(city)

## Definieer functies voor het uitvoeren van de preprocessing

Twee functies die de stappen gaan uitvoeren

	# Voor de tokenization en andere preprocessing stappen
	def spacy_prepro(text):
	    spacy_removal  = ['ADV','PRON','CCONJ','PUNCT','DET','ADP','SPACE']
	    text_out = []
	    doc = nlp(text)
	    for token in doc:
		if not token.is_stop and token.is_alpha and len(token) > 3 and token.pos_ not in spacy_removal:
		    lemma = token.lemma_
		    lemma = lemma.lower()
		    text_out.append(lemma)
	    return text_out

	# Voor het verwijderen van enkele bekende regular expressions
	def sent_to_words(sentences):
	    for sent in sentences:
		sent = re.sub('\S*@\S*\s?', '', str(sent))  # remove emails
		sent = re.sub('\s+', ' ', str(sent))  # remove newline chars
		sent = re.sub("\'", "", str(sent))  # remove single quotes
		sent = re.sub('[^A-Za-zëï]+', ' ', str(sent)) # remove non-Dutch-alphabetical characters (ö, ä, ñ, etc.)
		#sent = re.sub('[^aeiouAEIOU]{5,}', ' ', str(sent)) #kijken of dit werkt

		sent = re.sub('[a]{3,}', ' ', str(sent)) # instanties met 3 of meer a achter elkaar weg
		sent = re.sub('[e]{3,}', ' ', str(sent)) # instanties met 3 of meer e achter elkaar weg
		sent = re.sub('[i]{2,}', ' ', str(sent)) # instanties met 2 of meer i achter elkaar weg
		sent = re.sub('[o]{3,}', ' ', str(sent)) # instanties met 3 of meer o achter elkaar weg
		sent = re.sub('[u]{3,}', ' ', str(sent)) # instanties met 3 of meer u achter elkaar weg

		# Implement additional  spacy preprocessing
		sent = spacy_prepro(sent)

		yield(sent)

## Output

Bekijk dan de output die we krijgen in de vorm van een lijst van lijsten tokens.

	# Selecteer hier een subset van dp_df['text'] om het sneller te maken
	dp_df_subset = dp_df['text']
	# Voor het invoeren van alle docs, vul gewoon dp_df['text'] in als argument
	datalist_prepro = list(sent_to_words(dp_df_subset))
	print(datalist_prepro)



