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

package org.flag4j.arrays.dense;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field.AbstractDenseFieldMatrix;
import org.flag4j.arrays.sparse.CooFieldMatrix;
import org.flag4j.arrays.sparse.CsrFieldMatrix;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A dense matrix whose data are {@link Field} elements.</p>
 *
 * <p>Field matrices have mutable data but fixed shape.</p>
 *
 * <p>A matrix is essentially equivalent to a rank 2 tensor but has some extended functionality and may have improved performance
 * for some ops.</p>
 *
 * @param <T> Type of the {@link Field field} element for the matrix.
 */
public class FieldMatrix<T extends Field<T>> extends AbstractDenseFieldMatrix<FieldMatrix<T>, FieldVector<T>, T> {


    /**
     * Creates a dense field matrix with the specified data and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(Shape shape, T[] entries) {
        super(shape, entries);
    }


    /**
     * Creates a dense field matrix with the specified data and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(int rows, int cos, T[] entries) {
        super(new Shape(rows, cos), entries);
    }


    /**
     * Creates a dense field matrix with the specified data and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(T[][] entries) {
        super(new Shape(entries.length, entries[0].length), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense field matrix with the specified data and filled with {@code filledValue}.
     *
     * @param shape Shape of this matrix.
     * @param fillValue Entries of this matrix.
     */
    public FieldMatrix(Shape shape, T fillValue) {
        super(shape, (T[]) new Field[shape.totalEntriesIntValueExact()]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a dense field matrix with the specified data and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(int rows, int cols, T[][] entries) {
        super(new Shape(rows, cols), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense field matrix with the specified data and filled with {@code filledValue}.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param fillValue Entries of this matrix.
     */
    public FieldMatrix(int rows, int cols, T fillValue) {
        super(new Shape(rows, cols), (T[]) new Field[rows*cols]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero data of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooFieldMatrix<T> makeLikeCooTensor(Shape shape, T[] entries, int[][] indices) {
        return makeLikeCooMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and data.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and data.
     */
    @Override
    public FieldMatrix<T> makeLikeTensor(Shape shape, T[] entries) {
        return new FieldMatrix<T>(shape, entries);
    }


    /**
     * Constructs a vector of similar type to this matrix with the given {@code data}.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of similar type to this matrix with the given {@code data}.
     */
    @Override
    public FieldVector<T> makeLikeVector(T... entries) {
        return new FieldVector<T>(entries);
    }


    /**
     * Constructs a sparse CSR matrix of similar type to this dense matrix.
     *
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Row pointers of the CSR matrix.
     * @param colIndices Column indices of the non-zero data in the CSR matrix.
     *
     * @return A sparse CSR matrix with the specified shape and non-zero data.
     */
    @Override
    public CsrFieldMatrix<T> makeLikeCsrMatrix(Shape shape, T[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrFieldMatrix<T>(shape, entries, rowPointers, colIndices);
    }


    /**
     * Constructs a sparse COO matrix of similar type to this dense matrix.
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Row indices of the non-zero data in the COO matrix.
     * @param colIndices Column indices of the non-zero data in the COO matrix.
     *
     * @return A sparse COO matrix with the specified shape and non-zero data.
     */
    @Override
    public CooFieldMatrix<T> makeLikeCooMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        return new CooFieldMatrix<T>(shape, entries, rowIndices, colIndices);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooFieldMatrix<T> toCoo() {
        int rows = numRows;
        int cols = numCols;
        List<T> sparseEntries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                T val = data[rowOffset + j];

                if(!val.isZero()) {
                    sparseEntries.add(val);
                    rowIndices.add(i);
                    colIndices.add(j);
                }
            }
        }

        return new CooFieldMatrix<T>(shape, sparseEntries, rowIndices, colIndices);
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     * @return A sparse coo matrix equivalent to this matrix.
     * @see #toCoo()
     */
    public CsrFieldMatrix<T> toCsr() {
        // For simplicity convert to a COO matrix as an intermediate.
        return toCoo().toCsr();
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor with the same shape and data as this matrix.
     */
    @Override
    public FieldTensor<T> toTensor() {
        return new FieldTensor<>(new Shape(data.length), data.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified {@code newShape}.
     *
     * @param newShape Shape of the tensor. Can be any rank but must be broadcastable to the shape of this matrix.
     *
     * @return A tensor with the specified {@code newShape} and the same data as this matrix.
     */
    @Override
    public FieldTensor<T> toTensor(Shape newShape) {
        return new FieldTensor<>(newShape, data.clone());
    }


    /**
     * Constructs an identity matrix of the specified size.
     *
     * @param size Size of the identity matrix.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified size is less than 1.
     * @see #I(Shape, Field)
     * @see #I(int, int, Field)
     */
    public static FieldMatrix I(int size, Field fieldValue) {
        return I(size, size, fieldValue);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param numRows Number of rows in the identity-like matrix.
     * @param numCols Number of columns in the identity-like matrix.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int, Field)
     * @see #I(Shape, Field)
     */
    public static FieldMatrix I(int numRows, int numCols, Field fieldValue) {
        return I(new Shape(numRows, numCols), fieldValue);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param shape The shape of the identity-like matrix to construct.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int, Field)
     * @see #I(Shape, Field)
     */
    public static FieldMatrix I(Shape shape, Field fieldValue) {
        Field[] identityValues = new Field[shape.totalEntriesIntValueExact()];
        Arrays.fill(identityValues, (Field) fieldValue.getZero());
        Field one = (Field) fieldValue.getOne();

        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int i=0, stop=Math.min(rows, cols); i<stop; i++)
            identityValues[i*cols + i] = one;

        return new FieldMatrix(shape, identityValues);
    }


    /**
     * <p>Computes the matrix multiplication of this matrix with itself {@code n} times. This matrix must be square.</p>
     *
     * <p>For large {@code n} values, this method <i>may</i> significantly more efficient than calling
     * {@code #mult(Matrix) this.mult(this)} {@code n} times.</p>
     * @param n Number of times to multiply this matrix with itself. Must be non-negative.
     * @return If {@code n=0}, then the identity
     */
    public FieldMatrix<T> pow(int n) {
        ValidateParameters.ensureSquare(shape);
        ValidateParameters.ensureNonNegative(n);

        // Check for some quick returns.
        if (n == 0) return I(numRows, data[0]);
        if (n == 1) return copy();
        if (n == 2) return this.mult(this);

        FieldMatrix<T> result = I(numRows, data[0]);  // Start with identity matrix.
        FieldMatrix<T> base = this;

        // Compute the matrix power efficiently using an "exponentiation by squaring" approach.
        while(n > 0) {
            // If n is odd.
            if((n & 1) == 1)  result = result.mult(base);

            base = base.mult(base);  // Square the base.
            n >>= 1;  // Divide n by 2 (bitwise right shift).
        }

        return result;
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link FieldMatrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldMatrix<T> src2 = (FieldMatrix<T>) object;

        return shape.equals(src2.shape) && Arrays.equals(data, src2.data);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(data);

        return hash;
    }


    /**
     * Gets a row of the matrix formatted as a human-readable string.
     * @param rowIndex Index of the row to get.
     * @param columnsToPrint List of column indices to print.
     * @param maxWidths List of maximum string lengths for each column.
     * @return A human-readable string representation of the specified row.
     */
    private String rowToString(int rowIndex, List<Integer> columnsToPrint, List<Integer> maxWidths) {
        StringBuilder sb = new StringBuilder();

        // Start the row with appropriate bracket.
        sb.append(rowIndex > 0 ? " [" : "[");

        // Loop over the columns to print.
        for (int i = 0; i < columnsToPrint.size(); i++) {
            int colIndex = columnsToPrint.get(i);
            String value;
            int width = PrintOptions.getPadding() + maxWidths.get(i);

            if (colIndex == -1) // Placeholder for truncated columns.
                value = "...";
            else
                value = StringUtils.ValueOfRound(this.get(rowIndex, colIndex), PrintOptions.getPrecision());

            if (PrintOptions.useCentering())
                value = StringUtils.center(value, width);

            sb.append(String.format("%-" + width + "s", value));
        }

        // Close the row.
        sb.append("]");

        return sb.toString();
    }


    /**
     * Generates a human-readable string representing this matrix.
     * @return A human-readable string representing this matrix.
     */
    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        if (data.length == 0) {
            result.append("[]"); // No data in this matrix.
        } else {
            int numRows = this.numRows;
            int numCols = this.numCols;

            int maxRows = PrintOptions.getMaxRows();
            int maxCols = PrintOptions.getMaxColumns();

            int rowStopIndex = Math.min(maxRows - 1, numRows - 1);
            boolean truncatedRows = maxRows < numRows;

            int colStopIndex = Math.min(maxCols - 1, numCols - 1);
            boolean truncatedCols = maxCols < numCols;

            // Build list of column indices to print
            List<Integer> columnsToPrint = new ArrayList<>();
            for (int j = 0; j < colStopIndex; j++)
                columnsToPrint.add(j);

            if (truncatedCols) columnsToPrint.add(-1); // Use -1 to indicate '...'.
            columnsToPrint.add(numCols - 1); // Always include the last column.

            // Compute maximum widths for each column
            List<Integer> maxWidths = new ArrayList<>();
            for (Integer colIndex : columnsToPrint) {
                int maxWidth;
                if (colIndex == -1)
                    maxWidth = 3; // Width for '...'.
                else
                    maxWidth = PrettyPrint.maxStringLength(getCol(colIndex).data, rowStopIndex + 1);

                maxWidths.add(maxWidth);
            }

            // Build the rows up to the stopping index.
            for (int i = 0; i < rowStopIndex; i++) {
                result.append(rowToString(i, columnsToPrint, maxWidths));
                result.append("\n");
            }

            if (truncatedRows) {
                // Print a '...' row to indicate truncated rows.
                int totalWidth = maxWidths.stream().mapToInt(w -> w + PrintOptions.getPadding()).sum();
                String value = "...";

                if (PrintOptions.useCentering())
                    value = StringUtils.center(value, totalWidth);

                result.append(String.format(" [%-" + totalWidth + "s]\n", value));
            }

            // Append the last row.
            result.append(rowToString(numRows - 1, columnsToPrint, maxWidths));
        }

        result.append("]");

        return result.toString();
    }
}
