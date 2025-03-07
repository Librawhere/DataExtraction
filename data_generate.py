import json
import random
import numpy as np


def sample_name(n) -> list:

    first_names = []
    first_weight = []
    last_names = []
    last_weight = []

    # We consider the frequency of the name and use it as weights when we sample
    with open(r'./name/first.csv', 'r', encoding='utf-8-sig') as f:
        ds = f.readlines()
        for d in ds:
            d_ = d.strip().split('"')
            first_weight.append(d_[1].replace(',', ''))
            first_weight.append(d_[3].replace(',', ''))
            d_1 = d_[0].split(',')[1]
            d_2 = d_[2].replace(',', '')
            first_names.append(d_1)
            first_names.append(d_2)

    first_weight = np.array([int(i) for i in first_weight])
    first_weight = first_weight/sum(first_weight)
    random_firstname = random.choices(first_names, first_weight, k=n)

    with open(r'./name/Names_2010Census_Top1000.csv', 'r', encoding='utf-8') as f:
        ds = f.readlines()
        for d in ds[3:-4]:
            d_ = d.strip().split('"')
            last_weight.append(d_[1].replace(',', ''))
            last_names.append(d_[0].split(',')[0].capitalize())

    last_weight = np.array([int(i) for i in last_weight])
    last_weight = last_weight / sum(last_weight)
    random_lastname = random.choices(last_names, last_weight, k=n)

    name_lists = [random_firstname[i] + ' ' + random_lastname[i] for i in range(n)]

    return name_lists


# prescriptions.csv, D_ICD_DIAGNOSES.csv
# m is the number of diagnosis, n is the number of medicine
def sample_digmed(m, n):

    diags = []
    meds = []

    with open(r'./mimiciii/D_ICD_DIAGNOSES.csv', 'r', encoding='utf-8') as f:
        ds = f.readlines()
        for d in ds[1:]:
            d_ = d.strip().split('"')
            diags.append(d_[-2])

    with open(r'./mimiciii/PRESCRIPTIONS.csv', 'r', encoding='utf-8') as f:
        ds = f.readlines()
        for d in ds[1:]:
            d_ = d.strip().split(',')
            if d_[8] != '':
                meds.append((d_[1], d_[8].replace('"', '')))

    random_diags = random.choices(diags, weights=None, k=m)
    random_meds = random.choices(meds, weights=None, k=n)

    return [random_diags, random_meds]


def create_data(name, digmed, n):

    diags, meds = digmed

    # Prompts: A is diagnosed with B disease.
    lists_digs = [
        'Doctors have identified B# in A#. ',
        'A# has been told they have B#. ',
        'Medical tests confirm that A# has B#. ',
        'A# is suffering from B#, as per the diagnosis. ',
        'The presence of B# has been detected in A#. ',
        'A#’s condition has been classified as B#. ',
        'A# has been found to be battling B#. ',
        'A# has received a diagnosis of B#. ',
        'It has been determined that A# is affected by B#. ',
        'A#’s medical results indicate B#. '
    ]

    # Truncation of the diagnosis
    lists_digs_prompts = [
        'Doctors have identified B# in',
        'A# has been told they have',
        'Medical tests confirm that A# has',
        'A# is suffering from',
        'The presence of B# has been detected in',
        'A#’s condition has been classified as',
        'A# has been found to be battling',
        'A# has received a diagnosis of',
        'It has been determined that A# is affected by',
        'A#’s medical results indicate'
    ]

    # Prompts: Rearrange the follow sentence using different words or expressions.
    # Give at least 5 examples. Do not confine to a certain format:
    # A is taking B medicine.
    lists_meds = [
        'A# has been put on B#.',
        'A# is following a treatment plan that includes B#. ',
        'B# is part of A#’s daily routine. ',
        'A# takes B# as advised by their doctor. '
        'A# has started using B# to manage their condition. ',
        'The doctor prescribed B# for A#, and they are taking it accordingly. ',
        'A# relies on B# for relief. ',
        'A# includes B# in their healthcare regimen. ',
        'B# is what A# is using for treatment. ',
        'A# has been instructed to take B# regularly. '
    ]

    # truncation of the prescription
    lists_meds_prompts = [
        'A# has been put on',
        'A# is following a treatment plan that includes',
        'B# is part of',
        'A# takes'
        'A# has started using',
        'The doctor prescribed B# for',
        'A# relies on',
        'A# includes',
        'B# is what',
        'A# has been instructed to take'
    ]

    lens_diag = len(lists_digs)
    lens_med = len(lists_meds)

    pt_inferences = []

    with open('dataset/data.json', 'w', encoding='utf-8') as f:
        for _ in range(n):
            infer_diag_idx = random.randint(0, lens_diag-1)
            temp_diag = lists_digs[infer_diag_idx]     # template for diagnosis
            infer_diag_prompt = lists_digs_prompts[infer_diag_idx]

            infer_med_idx = random.randint(0, lens_med - 1)
            temp_med = lists_meds[infer_med_idx]      # template for medicine
            infer_med_prompt = lists_meds_prompts[infer_med_idx]

            # sample randomly from the name list for the diagnosis
            name_diag = random.choice(name)  # names for diagnosis
            diag = random.choice(diags)      # diagnosis

            # there is patient id in the prescription, so we use the id to sample from the name lists
            med_ = random.choice(meds)
            name_med = name[int(med_[0])//200]    # name for medicine
            med = med_[1]                         # medicine

            # save the truncated prompts for testing
            if 'A#' in infer_diag_prompt:
                sen1_pt = infer_diag_prompt.replace('A#', name_diag)
                label1 = diag
            else:
                sen1_pt = infer_diag_prompt.replace('B#', diag)
                label1 = name_diag
            pt_inferences.append([sen1_pt, label1])

            if 'A#' in infer_med_prompt:
                sen2_pt = infer_med_prompt.replace('A#', name_med)
                label2 = med
            else:
                sen2_pt = infer_med_prompt.replace('B#', med)
                label2 = name_med
            pt_inferences.append([sen2_pt, label2])

            # the following random.random() is to connect some of the sentences randomly to form
            # a longer paragraph, which is more aligned with the real situation. The number 0.7 is
            # just an empirical number
            sen1 = temp_diag.replace('A#', name_diag)
            if random.random() > 0.7:
                sen1 = sen1.replace('B#', diag) + '\n'
            else:
                sen1 = sen1.replace('B#', diag)

            sen2 = temp_med.replace('A#', name_med)
            if random.random() > 0.7:
                sen2 = sen2.replace('B#', med) + '\n'
            else:
                sen2 = sen2.replace('B#', med)

            f.write(sen1)
            f.write(sen2)

    with open('dataset/pt_inference.json', 'w', encoding='utf-8') as f:
        json.dump(pt_inferences, f)


if __name__ == '__main__':
    names = sample_name(500)
    digmeds = sample_digmed(1000, 1000)
    create_data(names, digmeds, 500)











