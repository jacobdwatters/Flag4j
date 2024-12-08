/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4j.io;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.backend.AbstractTensor;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.exceptions.Flag4jParsingException;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.function.Function;


/**
 * <p>Instances of this class may be used to parse a file containing matrix or vector data in
 * <a href="https://math.nist.gov/MatrixMarket/index.html">Matrix Market Exchange</a> Format.
 *
 * <p>Currently, this class only supports the following subset of the Matrix Market Format:
 * <ul>
 *   <li>Object: matrix</li>
 *   <li>Format: coordinate, array</li>
 *   <li>MMField: real, integer, complex</li>
 *   <li>MMSymmetry: general</li>
 * </ul>
 * <p>Headers must be of the form:
 *
 * <pre>
 * {@code %%MatrixMarket matrix <coordinate | array> <real | integer | complex> general}
 * </pre>
 *
 * @see MatrixMarketHeader
 * @see org.flag4j.io.MatrixMarketWriter
 */
public class MatrixMarketReader {

    /**
     * Error message for when a matrix with unsupported symmetry is attempted to be loaded.
     */
    private static final String UNSUPPORTED_SYMMETRY_ERR = "Unsupported symmetry found in Matrix Market Format file header: %s.";

    /**
     * Storage for header tokens of the Matrix Market file.
     */
    private MatrixMarketHeader header;
    /**
     * Current line from the buffered reader.
     */
    private String currLine;
    /**
     * Line number of the current line.
     */
    private int currLineNumber;
    /**
     * Storage for data from Matrix Market file.
     */
    private AbstractTensor<?, ?, ?> mat;
    /**
     * Buffered reader for reading Matrix Market file.
     */
    private BufferedReader reader;


    /**
     * Searches the file for the next non-empty line which is stored in {@link #currLine}.
     * @throws IOException If an I/O error occurs.
     */
    private void findNextNonemptyLine() throws IOException {
        do {
            currLine = reader.readLine();
            currLineNumber++;
        } while(currLine != null && currLine.isBlank());
    }


    /**
     * Parses the dimensions of a matrix.
     * @param expectedTokens The expected number of tokens on the current line.
     * @return The tokens as an array of integers.
     */
    private int[] parseDimensions(int expectedTokens) {
        String[] tokens = currLine.trim().split("\\s+");

        if(tokens.length != expectedTokens) {
            throw new Flag4jParsingException("Expected " + expectedTokens + " values for matrix shape/non-zero data but " +
                    "found " + tokens.length + ".");
        }

        int[] dimensions = new int[expectedTokens];
        for(int i = 0; i < expectedTokens; i++)
            dimensions[i] = Integer.parseInt(tokens[i]);

        return dimensions;
    }


    /**
     * <p>Loads Matrix Market data from the file specified on instantiation of this {@code MatrixMarketReader}.
     *
     * <p>Note, the value returned by this method is of type {@link AbstractTensor}. It is recommended to cast object to the desired
     * matrix or vector type. For instance:
     * <pre>{@code
     *      MatrixMarketReader parser = new MatrixMarketReader("some_file.mtx");
     *      Matrix matrix;
     *      var mm = parser.load();
     *
     *      try {
     *          matrix = (Matrix) mm;
     *      } catch(ClassCastException e) {
     *          // Handel exception.
     *      }
     * }</pre>
     * @param fileName The name of the Matrix Market Format file to read.
     * @return A matrix or vector containing the data in the specified Matrix Market file.
     * @throws IOException If an I/O error occurs.
     * @throws Flag4jParsingException If the Matrix Market Format file cannot be parsed.
     */
    public synchronized AbstractTensor<?, ?, ?> read(String fileName) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            this.reader = reader;
            currLine = reader.readLine();
            currLineNumber = 1;

            header = MatrixMarketHeader.parseHeader(currLine);
            MatrixMarketHeader.MMObject object = header.object();

            // Pass over all comments.
            do {
                findNextNonemptyLine();
            } while(currLine != null && currLine.startsWith("%"));

            if(currLine == null)
                throw new Flag4jParsingException("No lines found after header of Matrix Market Format file.");

            // If the supported objects are ever extend, this will need to be updated.
            if(object == MatrixMarketHeader.MMObject.MATRIX) {
                loadMatrix();
            } else {
                throw new Flag4jParsingException(
                        "Unsupported object type found in Matrix Market Format file header: " + object + ".");
            }

            return mat;
        }
    }


    /**
     * Loads a matrix object from a Matrix Market Format file.
     * @throws IOException If an I/O error occurs.
     */
    private void loadMatrix() throws IOException {
        MatrixMarketHeader.MMObject object = header.object();
        MatrixMarketHeader.MMFormat format = header.format();
        MatrixMarketHeader.MMField field = header.field();
        MatrixMarketHeader.MMSymmetry symmetry = header.symmetry();

        if(format == MatrixMarketHeader.MMFormat.ARRAY) {
            if(field == MatrixMarketHeader.MMField.REAL || field == MatrixMarketHeader.MMField.INTEGER) {
                // TODO: Once primitive int matrices are implemented, have special case for integers.
                if(symmetry == MatrixMarketHeader.MMSymmetry.GENERAL)
                    loadDenseRealMatrix();
                else
                    throw new Flag4jParsingException(String.format(UNSUPPORTED_SYMMETRY_ERR, symmetry));
            } else if(field == MatrixMarketHeader.MMField.COMPLEX) {
                if(symmetry == MatrixMarketHeader.MMSymmetry.GENERAL)
                    loadDenseComplexMatrix();
                else
                    throw new Flag4jParsingException(String.format(UNSUPPORTED_SYMMETRY_ERR, symmetry));
            } else {
                throw new Flag4jParsingException("Unsupported field found in Matrix Market Format file header: " + field + ".");
            }
        } else if(format == MatrixMarketHeader.MMFormat.COORDINATE) {
            if(field == MatrixMarketHeader.MMField.REAL || field == MatrixMarketHeader.MMField.INTEGER) {
                // TODO: Once primitive int matrices are implemented, have special case for integers.
                if(symmetry == MatrixMarketHeader.MMSymmetry.GENERAL)
                    loadCooRealMatrix();
                else
                    throw new Flag4jParsingException(String.format(UNSUPPORTED_SYMMETRY_ERR, symmetry));
            } else if(field == MatrixMarketHeader.MMField.COMPLEX) {
                if(symmetry == MatrixMarketHeader.MMSymmetry.GENERAL)
                    loadCooComplexMatrix();
                else
                    throw new Flag4jParsingException(String.format(UNSUPPORTED_SYMMETRY_ERR, symmetry));
            } else {
                throw new Flag4jParsingException("Unsupported field found in Matrix Market Format file header: " + field + ".");
            }
        } else {
            throw new Flag4jParsingException("Unsupported Format found in Matrix Market Format file header: " + format + ".");
        }
    }


    /**
     * Loads a real dense matrix from a Matrix Market Format file.
     * @throws IOException If an I/O error occurs.
     */
    private void loadDenseRealMatrix() throws IOException {
        Pair<Shape, List<Double>> matData = loadDenseMatrix(Double::parseDouble);
        Shape shape = matData.first();
        double[] data = ArrayUtils.fromDoubleList(matData.second());

        // Transpose to account for column major ordering in Matrix Market Format.
        mat = new Matrix(shape.swapAxes(0, 1), data).T();
    }


    /**
     * Loads a complex dense matrix from a Matrix Market Format file.
     * @throws IOException If an I/O error occurs.
     */
    private void loadDenseComplexMatrix() throws IOException {
        Function<String, Complex128> parseFunction = (String line) -> {
            StringTokenizer tokenizer = new StringTokenizer(currLine);
            double re = Double.parseDouble(tokenizer.nextToken());
            double im = Double.parseDouble(tokenizer.nextToken());

            if(tokenizer.hasMoreTokens())
                throw new Flag4jParsingException("Expecting two values for complex entries.");

            return new Complex128(re, im);
        };

        Pair<Shape, List<Complex128>> matData = loadDenseMatrix(parseFunction);
        Shape shape = matData.first();
        Complex128[] data = matData.second().toArray(new Complex128[0]);

        // Transpose to account for column major ordering in Matrix Market Format.
        mat = new CMatrix(shape.swapAxes(0, 1), data).T();
    }


    /**
     * Loads a real COO matrix from a Matrix Market Format file.
     * @throws IOException If an I/O error occurs.
     */
    private void loadCooRealMatrix() throws IOException {
        Function<StringTokenizer, Double> parseFunction =
                (StringTokenizer tokenizer) -> Double.parseDouble(tokenizer.nextToken());
        SparseMatrixData<Double> matData = loadCooMatrix(parseFunction);
        mat = new CooMatrix(matData.shape(), matData.data(), matData.rowData(), matData.colData());
    }


    /**
     * Loads a complex COO matrix from a Matrix Market Format file.
     * @throws IOException If an I/O error occurs.
     */
    private void loadCooComplexMatrix() throws IOException {
        Function<StringTokenizer, Complex128> parseFunction = (StringTokenizer tokenizer) ->
            new Complex128(Double.parseDouble(tokenizer.nextToken()), Double.parseDouble(tokenizer.nextToken()));
        SparseMatrixData<Complex128> matData = loadCooMatrix(parseFunction);
        mat = new CooCMatrix(matData.shape(), matData.data(), matData.rowData(), matData.colData());
    }


    /**
     * Loads a dense matrix from a Matrix Market file.
     * @param parseFunction Function to parse string for a single entry to the field type {@link T}.
     * @return A {@link Pair} containing the {@link Shape shape} of the matrix and a list containing the entries of the matrix in
     * colum-major ordering.
     * @param <T> Type corresponding to the field from the Matrix Market file header.
     * e.g. real -> {@code T} is type {@link Double}, complex -> {@code T} is type {@link Complex128}.
     */
    private <T> Pair<Shape, List<T>> loadDenseMatrix(Function<String, T> parseFunction) throws IOException {
        Shape shape = new Shape(parseDimensions(2));
        int rows = shape.get(0);
        int cols = shape.get(1);
        int lineLength = 2*rows;

        final int size = rows*cols;
        List<T> data = new ArrayList<T>(size);

        // MatrixMarket stores matrices in column major ordering.
        int idx = 0;
        for(int i=0; i<size; i++) {
            findNextNonemptyLine();
            if (currLine == null) {
                throw new Flag4jParsingException("Expecting " + size +
                        " entries in the Matrix Market file but found " + (i+1) + ".");
            }

            try{
                data.add(parseFunction.apply(currLine));
            } catch(NullPointerException e) {
                throw new Flag4jParsingException("Expecting " + size +
                        " entries in the Matrix Market Format file but found " + (i+1) + ".");
            }
        }

        findNextNonemptyLine();
        if(currLine != null) {
            throw new Flag4jParsingException("Found extra non-empty lines at the end of the Matrix Market Format file. " +
                    "The number of data lines does not match the number of columns specified: " + cols + ".");
        }

        return new Pair<>(shape, data);
    }


    /**
     * Loads a complex COO matrix from a Matrix Market Format file.
     * @param parseFunction Function to parse {@link StringTokenizer} for a single data entry to the field type {@link T}.
     * @return A {@link SparseMatrixData} object storing the shape, non-zero entries, row indices, and column indices of the COO
     * matrix.
     * @throws IOException If an I/O error occurs.
     * @param <T> Type corresponding to the field from the Matrix Market file header.
     * e.g. real &rarr; {@code T} is type {@link Double}, complex &rarr; {@code T} is type {@link Complex128}.
     */
    private <T> SparseMatrixData<T> loadCooMatrix(Function<StringTokenizer, T> parseFunction) throws IOException {
        int[] shape = parseDimensions(3);
        int rows = shape[0];
        int cols = shape[1];
        int nnz = shape[2];

        List<T> data = new ArrayList<>(nnz);
        List<Integer> rowIndices = new ArrayList<>(nnz);
        List<Integer> colIndices = new ArrayList<>(nnz);

        for(int i=0; i<nnz; i++) {
            findNextNonemptyLine();

            if (currLine == null) {
                throw new Flag4jParsingException("Expecting " + nnz +
                        " non-zero entries for coordinate Format but found " + (i+1) + ".");
            }

            StringTokenizer tokenizer = new StringTokenizer(currLine);

            int startTokens = tokenizer.countTokens();
            // Indices are one based in Matrix Market Format files so shift down by one.
            rowIndices.add(Integer.parseInt(tokenizer.nextToken()) - 1);
            colIndices.add(Integer.parseInt(tokenizer.nextToken()) - 1);
            data.add(parseFunction.apply(tokenizer));
            int endTokens = tokenizer.countTokens();

            if(tokenizer.hasMoreTokens()) {
                int expectedTokens = startTokens - endTokens;
                throw new Flag4jParsingException("Expecting exactly " + expectedTokens + " values in Matrix Market " +
                        "file for " + header.field() + " coordinate Format " +
                        "but found " + tokenizer.countTokens() + " on line " + currLineNumber + ".");
            }
        }

        findNextNonemptyLine();
        if(currLine != null) {
            throw new Flag4jParsingException("Found extra non-empty lines at the end of the Matrix Market Format file. " +
                    "The number of  data lines does not match the number of non-zero entries specified: " + cols + ".");
        }

        return new SparseMatrixData<T>(new Shape(rows, cols), data, rowIndices, colIndices);
    }
}
