import os
import datasets.biocause.spacies.constants as constants
import datasets.biocause.spacies.create_files as create_files
from datasets.biocause.spacies.document import Document, EntityMention, TriggerMention, Event

# parse the biocause dataset using scispacy
# python=3.6.13
# spacy=3.0.7
# scispacy=0.4.0
class BioCausePreprocessor:
    def __init__(self,
                 root_path=None,
                 use_sec_entities=False,
                 bind_renaming='no',
                 keep_entity_tokens=True,
                 keep_orphan_entities=False,
                 encoding='mt.1',
                 multihead=True,
                 masking='no'):
        self.root_path = root_path
        self.use_sec_entities = use_sec_entities
        self.bind_renaming = bind_renaming
        self.keep_orphan_entities = keep_orphan_entities
        self.keep_entity_tokens = keep_entity_tokens
        self.encoding = encoding
        self.multihead = multihead
        self.masking = masking
        self.use_trigger = ['Negative_regulation', 'Localization', 'Process', 'Phosphorylation',
                            'Gene_expression', 'Binding', 'Positive_regulation', 'Causality',
                            'Transcription', 'Protein_catabolism', 'Regulation']
        self.doc_ids = self.get_doc_ids_from_dir(path=self.root_path)
        self.generate_data()

    def get_doc_ids_from_dir(self, path):
        ids = set()
        # Get only the files (no directories)
        files = [f for f in os.listdir(path) if os.path.isfile(
            os.path.join(path, f))]

        # Add only the filenames (no extensions) without duplicates
        for file in files:
            filename, extension = os.path.splitext(file)
            if filename not in constants.EXCL_FILENAMES:
                ids.add(filename)

        return sorted(list(ids))

    def generate_data(self):
        documents = []
        trigger_types = set()
        # Create a list of document objects containing the annotations
        for doc_id in self.doc_ids:
            document = self.parse_document_files(
                self.root_path, doc_id, self.use_sec_entities,
                self.bind_renaming)
            for key, event in document.events.items():
                trigger_types.add(event.type_)
            documents.append(document)

        # Create the file of encoded events for the documents in the split
        create_files.create_annotations(
            documents, self.use_sec_entities,
            self.keep_entity_tokens, self.keep_orphan_entities, self.encoding,
            self.multihead, self.masking)

    def parse_mention(self, line):
        m_id = line[0]
        raw_info = line[1].split(" ")  # the middle part, space-separated
        m_type = raw_info[0]
        m_start = int(raw_info[1])
        m_end = int(raw_info[2])
        m_text = line[2]

        return m_id, m_type, m_start, m_end, m_text

    def parse_event(self, line):
        e_id = line[0]
        participants = line[1].split()  # list of [ARG_TYPE]:[ID/E_ID]
        is_event_trigger = True
        e_edge_types = []
        e_end_ids = []

        for participant in participants:
            arg_label, participant_id = participant.split(":")

            # The first participant is the event trigger, i.e., [TYPE]:[TID]
            if is_event_trigger:
                e_type = arg_label  # the event type, e.g., Binding
                e_start_id = participant_id  # the ID of the source trigger
                is_event_trigger = False  # switch the flag off for arguments

            # From the second onwards, there are arguments, i.e., [ETYPE]:[ID/EID]
            else:
                e_edge_type = arg_label  # the argument type, e.g., Theme
                e_end_id = participant_id  # the ID of the target event/entity
                e_edge_types.append(e_edge_type)
                e_end_ids.append(e_end_id)

        return e_id, e_type, e_start_id, e_edge_types, e_end_ids

    def rename_binding_events(self, events, triggers, bind_renaming):
        # last_start_id = None
        for event in events.items():
            if event[1].type_ == "Binding":
                is_multi_trigger = False
                is_multi_argument = False
                # is_multi_argument_partial = False
                # same_event_of_before = (last_start_id == event[1].start_id)

                # Check if there are multiple events centered on the trigger
                if event[1].num > 1:
                    is_multi_trigger = True

                # Check if there are multiple arguments for the event
                list_of_args = event[1].edge_types
                count = 0
                # print(event[1].start_id, same_event_of_before, list_of_args)
                for arg in list_of_args:
                    if arg.startswith("Theme"):
                        count += 1
                if count > 1:
                    is_multi_argument = True

                # Rename the trigger types (and optionally the triggers)
                if is_multi_trigger:
                    # CASE: K
                    if is_multi_argument:
                        triggers[event[1].start_id].type_ = "Binding1"
                        if not bind_renaming.endswith("only_tri"):
                            event[1].type_ = "Binding1"
                    # CASE: N
                    else:
                        triggers[event[1].start_id].type_ = "BindingN"
                        if not bind_renaming.endswith("only_tri"):
                            event[1].type_ = "BindingN"
                else:
                    triggers[event[1].start_id].type_ = "Binding1"
                    if not bind_renaming.endswith("only_tri"):
                        event[1].type_ = "Binding1"

                # last_start_id = event[1].start_id

    def get_index_positions(self, list, element):
        indexes = []
        index_position = 0
        while True:
            try:
                index_position = list.index(element, index_position)
                indexes.append(index_position)
                index_position += 1
            except ValueError as error:
                break

        return indexes

    def parse_document_files(self, folder, doc_id, use_sec_entities, bind_renaming):
        txt_path = os.path.join(folder, doc_id + constants.EXT_FILES["txt"])
        ann_path = os.path.join(folder, doc_id + '.ann')
        paragraphs = []
        entities = {}
        triggers = {}
        events = {}

        # [.TXT]: Store the list of paragraphs in the document
        with open(txt_path, mode="r", encoding="utf-8") as f:
            for line in f:
                paragraphs.append(line.rstrip("\n"))
        org_txt = open(txt_path, mode="r", encoding="utf-8").read()
        # [.A1]: Store the list of entities which BioNLP-ST format is:
        #    [EID]\t[TYPE] [START_CHAR] [END_CHAR]\t[TEXT]
        with open(ann_path, mode="r", encoding="utf-8") as f:
            tmp_triggers = []
            tmp_events = []
            for line in f:
                spl_line = line.rstrip("\n").split("\t")
                if spl_line[0].startswith('T'):
                    if len(spl_line) == 3:
                        e_id, e_type, e_start, e_end, e_text = self.parse_mention(spl_line)
                        if e_type in self.use_trigger:
                            trigger = TriggerMention(e_id, e_type, e_start, e_end, e_text)
                            triggers[e_id] = trigger
                        else:
                            entity = EntityMention(e_id, e_type, e_start, e_end, e_text)
                            entities[e_id] = entity
                    elif len(spl_line) == 2:
                        Ttype, start, end = spl_line[-1].split()
                        start, end = int(start), int(end)
                        T_text = org_txt[start:end]
                        spl_line.append(T_text)
                        e_id, e_type, e_start, e_end, e_text = self.parse_mention(spl_line)
                        if e_type in self.use_trigger:
                            trigger = TriggerMention(e_id, e_type, e_start, e_end, e_text)
                            triggers[e_id] = trigger
                        else:
                            entity = EntityMention(e_id, e_type, e_start, e_end, e_text)
                            entities[e_id] = entity
                elif spl_line[0].startswith("E"):
                    if len(spl_line) == 2:
                        attributes = self.parse_event(spl_line)
                        id_, type_, start_id, arg_types, end_ids = attributes
                        tmp_triggers.append(start_id)
                        num = tmp_triggers.count(start_id)
                        tmp_events.append(id_)
                        # If there are previous Binding events centered on the
                        # same trigger, update their "num" to optionally renaming
                        # them afterwards (@TODO: Generalize to Theme+ events)
                        if (num > 1) and (type_ == "Binding"):
                            # Get the indexes of all occurrences of the trigger
                            trigger_pos_idx = self.get_index_positions(
                                tmp_triggers, start_id)

                            # For each index, update the "num" of the event
                            # The last one is the current (already correct)
                            for i in range(len(trigger_pos_idx) - 1):
                                events[tmp_events[trigger_pos_idx[i]]].num = num

                        event = Event(
                            id_, type_, start_id, num, arg_types, end_ids)
                        events[id_] = event
                    elif len(spl_line) == 3:
                        print()
                    else:
                        print()

        # Rename the Binding events
        if self.bind_renaming != "no":
            self.rename_binding_events(events, triggers, bind_renaming)

        # Store all the information into a Document object
        document = Document(doc_id, paragraphs, entities, triggers, events)

        return document

BioCausePreprocessor = BioCausePreprocessor(root_path='../../../sources/biocause',
                                            use_sec_entities=False,
                                            bind_renaming='no')