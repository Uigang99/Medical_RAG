"""Versioned, example-free refinements; original v1/v2 remain immutable."""
from copy import deepcopy
from medrag.kev_chunk_relations import make_job as original_job, TASKS
from medrag.training.graph_input_data import digest

DEFINITIONS = {
    'v3': {
        'EXPLAINS': (
            'Does SOURCE supply an explanation for something stated in TARGET? '
            'Find a particular target statement or term, then a source passage that '
            '(a) defines that term, or (b) states why or how that statement holds. '
            'YES requires this specific explanation, not a general subject match. '
            'The source can explain just one part of the target; the target may already '
            'mention the same mechanism briefly. A more explicit account of how it works '
            'still qualifies. Repetition without explanatory information does not. '
            'An example or category list alone is not an explanation. The connection '
            'must follow from the two texts without inventing a causal premise.'),
        'PROCESS_PRECEDES': (
            'Is there an earlier stage in SOURCE and a later, different stage in TARGET '
            'of the same process? Look for ONE earlier/later stage pair, not an ordering '
            'of the whole documents. YES when the texts establish that an action produces '
            'an intermediate state which is then acted on, an initiation leads to a '
            'subsequent change, or a stated procedure moves from one step to another. '
            'The chunks may overlap, and each may also mention stages in the other chunk. '
            'This does not cancel a valid earlier/later pair in the requested direction. '
            'General descriptions need not come from the same document to describe the '
            'same process. Separate specific patients, experiments, or studies are NOT '
            'one sequence unless their continuity is established in these texts. '
            'A shared topic, two static properties, or chronology guessed from outside '
            'knowledge is NO. Do not infer the sequence just from a usual workflow.'),
        'CONTEXTUAL_CONTRAST': (
            'Do the chunks describe different alternatives or circumstances for the SAME '
            'concrete issue? Identify the common issue (what phenomenon, task, decision, '
            'or outcome is being discussed), and then a relevant difference stated by '
            'the texts. Different mechanisms for the same outcome, methods for the same '
            'task, or states/stages for the same management issue qualify. YES does not '
            'require disagreement, opposite results, or an explicit comparison sentence. '
            'Both descriptions can be true and useful. NO if there are only different '
            'facts under a broad medical topic, a general statement plus its matching '
            'example, or different patients without a substantive comparison axis. '
            'Do not invent a shared purpose, identity, or difference.'),
    },
}

DEFINITIONS['v4'] = {
    'EXPLAINS': (
        'Does information in SOURCE explain a statement or term found in TARGET? '
        'First identify something actually stated in TARGET. Then ask whether SOURCE '
        'provides its reason, mechanism, or the meaning of a term used there. A definition '
        'or a text-supported why/how account is YES, even if it explains only a small '
        'part of TARGET. SOURCE need not introduce the term before TARGET does. '
        'The direction is explanation provider SOURCE -> explained content TARGET, '
        'not cause -> effect mentioned inside one text. Repeating a fact, giving an '
        'example, or listing another fact on the topic without explaining it is NO. '
        'Do not add an unstated mechanism or assume that separate cases are identical.'),
    'PROCESS_PRECEDES': (
        'Can one stage described in SOURCE be placed before a different stage described '
        'in TARGET along a sequence established by these texts? Match the process and '
        'find an earlier/later stage pair. Either text can establish the connecting '
        'sequence. A stated causal chain (an initiating action, an intermediate change, '
        'then a consequence) or an ordered procedure qualifies. Overlapping descriptions '
        'of several stages do NOT force NO: choose the source earlier stage and target '
        'later stage, not an order between entire documents. The reverse direction may '
        'also be true when both chunks cover multiple stages. General accounts of the '
        'same process need not share a document. Separate specific studies/patients are '
        'not a single sequence without textual continuity. Static associations, mere '
        'topic similarity, and order requiring an unstated medical premise are NO.'),
    'CONTEXTUAL_CONTRAST': DEFINITIONS['v3']['CONTEXTUAL_CONTRAST'] + (
        ' Distinguish a substantive alternative from a difference in writing detail: '
        'a broad overview and a detailed account of the SAME mechanism are not contrasting '
        'alternatives just because the detailed account mentions more steps. Two listings '
        'of the same alternatives without differing comparison information are also NO.'),
}

# Keep the previously strong overlap definition.  The three other relations
# are revised independently so the effect of each can be inspected.
DEFINITIONS['v5'] = {
    'EXPLAINS': (
        'Decide whether SOURCE supplies a reason, mechanism, or definition for '
        'a particular statement or term actually present in TARGET. Identify '
        'that TARGET statement first. YES only when the SOURCE information '
        'answers WHY/HOW that statement holds or WHAT its term means, using '
        'facts stated in these chunks. The explanatory information must add '
        'something beyond repeating the TARGET fact: listing the same types, '
        'markers, outcomes, or recommendations is NO unless SOURCE also states '
        'the causal, functional, or definitional link. A more detailed account '
        'of a briefly stated mechanism can qualify. A temporal predecessor '
        'alone is not automatically an explanation. Never reverse SOURCE and '
        'TARGET or supply a missing medical premise.'),
    'PROCESS_PRECEDES': DEFINITIONS['v4']['PROCESS_PRECEDES'] + (
        ' Do not label YES merely because two documents mention the same '
        'clinical workflow or historical era: the particular earlier stage in '
        'SOURCE and later stage in TARGET must be distinguishable in the '
        'supplied text. If both directions seem true, each direction needs '
        'its own different-stage witness.'),
    'CONTEXTUAL_CONTRAST': (
        'Do A and B give substantively different alternatives, conditions, '
        'states, or positions for one identifiable comparison issue? YES if '
        'one can ask the SAME concrete question of both portions and obtain '
        'different answers grounded in the two chunks, even though both '
        'answers may be true. Different uses of the same intervention, '
        'different causes of the same stated outcome, and different management '
        'contexts can qualify. A broad topic alone is not the comparison '
        'issue. A general rule plus its matching example, repeated lists of '
        'the same alternatives, or extra detail about one mechanism are NO. '
        'Do not invent a common patient, event, task, or difference.'),
}

DEFINITIONS['v6'] = {
    'EXPLAINS': (
        'Does SOURCE explain something actually stated in TARGET? A definition '
        'of a target term, or a stated reason or mechanism for a target finding '
        'or action, is YES. One specific portion suffices; SOURCE may repeat '
        'the target fact as well, provided it also supplies the explanatory '
        'link. A list of the same examples, categories, findings, or outcomes '
        'without such a link is NO. Check the requested direction: the '
        'explaining content must be in SOURCE, not TARGET. Use only the two '
        'texts and ordinary paraphrase; do not supply a missing premise.'),
    'PROCESS_PRECEDES': DEFINITIONS['v4']['PROCESS_PRECEDES'],
    'CONTEXTUAL_CONTRAST': (
        'Do A and B describe meaningfully different alternatives or contexts '
        'for an identifiable common problem, phenomenon, decision, or outcome? '
        'YES when the chunks state different causes of the same phenomenon, '
        'different interventions for the same task, or different states or '
        'applications relevant to the same decision. They need not explicitly '
        'compare each other or disagree, and one chunk may contain a broader '
        'list while the other details one genuinely distinct alternative. '
        'NO when the only difference is added detail about the same alternative, '
        'repetition of the same list, a shared broad topic, or separate cases '
        'with no matched issue. Never invent the common problem or difference.'),
}


def make_job(pair, task, version):
    if version == 'v7':
        # Frozen task-wise selection, not an additional prompt rewrite.
        chosen = 'v4' if TASKS[task][1] in ('EXPLAINS', 'PROCESS_PRECEDES') else 'v2'
        return dict(make_job(pair, task, chosen), prompt_version=version)
    if version == 'v2':
        return dict(original_job(pair, task, 'v2'), prompt_version=version)
    if version not in DEFINITIONS:
        raise ValueError(version)
    original = original_job(pair, task, 'v2')
    job = deepcopy(original)
    _, relation, direction = TASKS[task]
    if relation in DEFINITIONS[version]:
        instruction = DEFINITIONS[version][relation]
        if direction and version not in ('v4', 'v5', 'v6'):
            source, target = ('A', 'B') if direction == 'A_TO_B' else ('B', 'A')
            instruction += f' For this decision SOURCE is chunk {source}; TARGET is chunk {target}.'
        job['body']['questions']['RELATION']['instructions'] = instruction
    if version in ('v4', 'v5', 'v6') and direction:
        source, target = ('A','B') if direction == 'A_TO_B' else ('B','A')
        job['body']['state'] = {'rules':job['body']['state']['rules'],
                               'SOURCE':pair[source]['text'], 'TARGET':pair[target]['text']}
    # Identical bodies intentionally have identical IDs, allowing exact cached reuse.
    job['id'] = original['id'] if job['body'] == original['body'] else digest(['kev27_refinement', job['body']])
    job['prompt_version'] = version
    return job
