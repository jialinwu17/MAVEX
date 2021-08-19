
import inflect
inflect = inflect.engine()

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def concept_to_sentence(concept):
    singular =  (inflect.singular_noun(concept[0]) == False)
    if singular:
        be_verb = 'is'
    else:
        be_verb = 'are'
    if concept[1] == 'RelatedTo':
        verb = be_verb + ' related to'
    elif concept[1] == 'HasContext':
        if singular:
            verb = 'has a context of'
        else:
            verb = 'have a context of'
    elif concept[1] == 'FormOf':
        verb = be_verb + ' a form of '
    elif concept[1] == 'IsA':
        verb = be_verb
    elif concept[1] == 'Synonym':
        verb = be_verb + ' a synonym of'
    elif concept[1] == 'DistinctFrom':
        verb = be_verb + ' distinct from'
    elif concept[1] == 'Antonym':
        verb = be_verb + ' an antonym of'
    elif concept[1] == 'NotHasProperty':
        if singular:
            verb = ' does not have property of'
        else:
            verb = ' do not have property of'
    elif concept[1] == 'CapableOf':
        verb = be_verb + ' capable of'
    elif concept[1] == 'DerivedFrom':
        verb = be_verb + ' derived from'
    elif concept[1] == 'EtymologicallyRelatedTo' or concept[1] == 'EtymologicallyDerivedFrom':
        verb = be_verb + ' etymologically related to'
    elif concept[1] == 'ExternalURL':
        verb = be_verb
    elif concept[1] == 'PartOf':
        if singular:
            verb = ' is a part of'
        else:
            verb = ' are parts of'
    elif concept[1] == 'MannerOf':
        verb = be_verb + ' manner of'
    elif concept[1] == 'Entails':
        verb = 'Entail'
        if singular:
            verb += 's'
    elif concept[1] == 'SimilarTo':
        verb = be_verb + ' similar to'
    elif concept[1] == 'InstanceOf':
        if singular:
            verb = ' is a instance of'
        else:
            verb = ' are instances of'
    elif concept[1] == 'HasA':
        if singular:
            verb = ' has'
        else:
            verb = ' have'
    elif concept[1] == 'language':
        verb = be_verb + ' in'
    elif concept[1] == 'MotivatedByGoal':
        verb = 'because'
    elif concept[1] == 'HasPrerequisite':
        verb = 'require'
        if singular:
            verb += 's'
    elif concept[1] == 'HasFirstSubevent':
        verb = 'require'
        if singular:
            verb += 's'
        verb += ' you to first'
    elif concept[1] == 'HasSubevent':
        verb = 'require'
        if singular:
            verb += 's'
        verb += ' you to'
    elif concept[1] == 'genus' or concept[1] == 'genre':
        if singular:
            verb = ' is a kind of'
        else:
            verb = ' are kinds of'
    elif concept[1] == 'UsedFor':
        verb = be_verb + ' used to'
    elif concept[1] == 'ReceivesAction':
        verb = 'can be'
    elif concept[1] == 'Causes':
        if singular:
            verb = 'causes'
        else:
            verb = 'cause'
    elif concept[1] == 'CausesDesire':
        if singular:
            verb = 'makes you want to'
        else:
            verb = 'make you want to'
    elif concept[1] == 'MadeOf':
        verb = be_verb + ' made of'
    elif concept[1] == 'capital':
        verb = 'has capital'
    elif concept[1] == 'DefinedAs':
        verb = 'can be defined as'
    elif concept[1] == 'NotDesires':
        if singular:
            verb = 'does not want to'
        else:
            verb = 'do not want to'
    elif concept[1] == 'CreatedBy':
        verb = be_verb + ' creaeted by'
    elif concept[1] == 'product':
        if singular:
            verb = 'produces'
        else:
            verb = 'produce'
    elif concept[1] == 'occupation':
        verb =  'is an occupation of'
    elif concept[1] == 'Desires':
        if singular:
            verb = 'wants to'
        else:
            verb = 'want to'
    elif concept[1] == 'AtLocation':
        if singular:
            verb = 'is at'
        else:
            verb = 'are at'
    elif concept[1] == 'LocatedNear':
        if singular:
            verb = 'is near'
        else:
            verb = 'are near'
    elif concept[1] == 'HasLastSubevent':
        verb = 'because you will'
    elif concept[1] == 'NotCapableOf':
        verb = be_verb + ' not capable of'
    elif concept[1] == 'influencedBy':
        verb = be_verb + ' influenced by'
    elif concept[1] == 'field':
        verb = 'works on the field of'
    elif concept[1] == 'knownFor':
        verb = be_verb + ' known for'
    elif concept[1] == 'leader':
        verb = be_verb + ' led by'
    elif concept[1] == 'SymbolOf':
        verb = be_verb + ' symbol of'
    elif concept[1] == 'HasProperty':
        if singular:
            verb = 'has property of'
        else:
            verb = 'have properties of'
    else:
        print('aaaaaaaaaaaaaaaaaaa')
        verb = be_verb
        print(concept[1])
    return '%s %s %s'%(concept[0], verb,concept[2])

