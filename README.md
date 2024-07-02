# Interest Rate Models Replication

This project aims to replicate the interest rate models explained in Brigo and Mercurio's book on interest rates using advanced Python structures such as factories and templates.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to deepen understanding of interest rate models by implementing them in Python. However, it also serves to me as practice to handle factories and advanced python designs.

 The models are based on the frameworks presented in Brigo and Mercurio's book, and the implementation focuses on using advanced Python programming techniques.

## Prerequisites

- Python 3.8 or higher
- Basic knowledge of Python and object-oriented programming
- Familiarity with interest rate modeling

## Installation

Clone the repository:

```bash
git clone https://github.com/belgrano9/IRMBook.git
cd IRMBook
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the directory containing the model scripts.
2. Run the scripts to see the models in action.

Example:

```bash
python models/hull_white_model.py
```

## Models

- **Hull-White Model**: Implements the Hull-White interest rate model using a factory pattern.
- **Vasicek Model**: Implements the Vasicek interest rate model with a template method.
- **Cox-Ingersoll-Ross (CIR) Model**: Implements the CIR model using a combination of factory and template patterns.

Each model is located in the `models` directory and follows a structured design pattern for clarity and extensibility.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
