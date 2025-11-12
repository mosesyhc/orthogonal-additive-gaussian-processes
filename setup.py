# +
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup
# -

setup(
    name="oak",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gpflow>=2.2.1",
        "pytest>=5.4.1",
        # "seaborn",
        # "tqdm>=4.44.1",
        # "tikzplotlib",
        "scikit-learn",
        "numpy",
        # "matplotlib<=3.7",
        # "seaborn",
        "IPython",
        "tensorflow==2.18.0",
        "s3fs>=0.4.0",
        "scikit-learn-extra>=0.2.0",
        "tensorflow_probability==0.25.0",
    ],
)
