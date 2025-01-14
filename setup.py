# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name='conformer',
    packages = find_packages(),
    version='latest',
    description='Convolution-augmented Transformer for Speech Recognition',
    author='Soohwan Kim',
    author_email='sh951011@gmail.com',
    url='https://github.com/sooftware/conformer',
    install_requires=[
        'numpy',
    ],
    keywords=['asr', 'speech_recognition', 'conformer', 'end-to-end'],
    python_requires='>=3.6'
)
