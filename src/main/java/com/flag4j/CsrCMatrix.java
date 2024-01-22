package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.sparse.ComplexSparseTensorBase;
import com.flag4j.io.PrintOptions;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.StringUtils;

import java.util.Arrays;

public class CsrCMatrix extends ComplexSparseTensorBase<CsrCMatrix, CMatrix, CsrMatrix> {

    /**
     * Row indices of the non-zero entries of the sparse matrix.
     */
    public final int[] rowPointers;
    /**
     * Column indices of the non-zero entries of the sparse matrix.
     */
    public final int[] colIndices;
    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;


    /**
     * Constructs a sparse matrix in CSR format with specified row-pointers, column indices and non-zero entries.
     * @param shape Shape of the matrix.
     * @param entries Non-zero entries for CSR matrix.
     * @param rowPointers Row pointers for CSR matrix.
     * @param colIndices Column indices for CSR matrix.
     */
    public CsrCMatrix(Shape shape, CNumber[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries.length, entries, rowPointers, colIndices);

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }


    /**
     * Constructs a sparse matrix in CSR format with specified row-pointers, column indices and non-zero entries.
     * @param shape Shape of the matrix.
     * @param entries Non-zero entries for CSR matrix.
     * @param rowPointers Row pointers for CSR matrix.
     * @param colIndices Column indices for CSR matrix.
     */
    public CsrCMatrix(Shape shape, double[] entries, int[] rowPointers, int[] colIndices) {
        super(shape, entries.length, new CNumber[entries.length], new int[colIndices.length], colIndices);

        ArrayUtils.copy2CNumber(entries, this.entries); // Copy entries from double array to CNumber array.

        this.rowPointers = rowPointers;
        this.colIndices = colIndices;
        numRows = shape.dims[0];
        numCols = shape.dims[1];
    }



    /**
     * Converts a sparse COO matrix to a sparse CSR matrix.
     * @param src COO matrix to convert. Indices must be sorted lexicographically.
     */
    public CsrCMatrix(CooCMatrix src) {
        super(src.shape.copy(),
                src.entries.length,
                new CNumber[src.entries.length],
                new int[src.numRows + 1],
                src.colIndices.clone()
        );

        rowPointers = this.indices[0];
        colIndices = this.indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];

        ArrayUtils.copy2CNumber(src.entries, entries); // Deep copy non-zero entries.

        // Copy the non-zero entries anc column indices. Count number of entries per row.
        for(int i=0; i<src.entries.length; i++) {
            rowPointers[src.rowIndices[i] + 1]++;
        }

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<src.numRows; i++) {
            rowPointers[i+1] += rowPointers[i];
        }
    }


    /**
     * Converts a real sparse COO matrix to a complex sparse CSR matrix.
     * @param src COO matrix to convert. Indices must be sorted lexicographically.
     */
    public CsrCMatrix(CooMatrix src) {
        super(src.shape.copy(),
                src.entries.length,
                new CNumber[src.entries.length],
                new int[src.numRows + 1],
                src.colIndices.clone()
        );

        rowPointers = this.indices[0];
        colIndices = this.indices[1];
        this.numRows = shape.dims[0];
        this.numCols = shape.dims[1];

        ArrayUtils.copy2CNumber(src.entries, entries); // Deep copy non-zero entries.

        // Copy the non-zero entries anc column indices. Count number of entries per row.
        for(int i=0; i<src.entries.length; i++) {
            rowPointers[src.rowIndices[i] + 1]++;
        }

        // Shift each row count to be greater than or equal to the previous.
        for(int i=0; i<src.numRows; i++) {
            rowPointers[i+1] += rowPointers[i];
        }
    }

    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #H}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CsrCMatrix hermTranspose() {
        return null;
    }

    /**
     * Computes the conjugate transpose of this tensor. In the context of a tensor, this swaps the first and last axes
     * and takes the complex conjugate of the elements along these axes. Same as {@link #hermTranspose()}.
     *
     * @return The complex transpose of this tensor.
     */
    @Override
    public CsrCMatrix H() {
        return null;
    }

    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     * @throws IllegalArgumentException  If the number of indices is not equal to the rank of this tensor.
     * @throws IndexOutOfBoundsException If any of the indices are not within this tensor.
     */
    @Override
    public CsrCMatrix set(CNumber value, int... indices) {
        return null;
    }

    /**
     * Simply returns a reference of this tensor.
     *
     * @return A reference to this tensor.
     */
    @Override
    protected CsrCMatrix getSelf() {
        return null;
    }

    /**
     * Checks if all entries of this tensor are close to the entries of the argument {@code tensor}.
     *
     * @param tensor Tensor to compare this tensor to.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if the argument {@code tensor} is the same shape as this tensor and all entries are 'close', i.e.
     * elements {@code a} and {@code b} at the same positions in the two tensors respectively satisfy
     * {@code |a-b| <= (atol + rtol*|b|)}. Otherwise, returns false.
     * @see #allClose(Object)
     */
    @Override
    public boolean allClose(CsrCMatrix tensor, double relTol, double absTol) {
        return false;
    }

    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     * @return A reference to this tensor.
     */
    @Override
    public CsrCMatrix set(double value, int... indices) {
        return null;
    }

    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @throws IllegalArgumentException If the axis is not positive or larger than the rank of this tensor.
     */
    @Override
    public CsrCMatrix flatten(int axis) {
        return null;
    }

    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CsrCMatrix add(CsrCMatrix B) {
        return null;
    }

    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(double a) {
        return null;
    }

    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix add(CNumber a) {
        return null;
    }

    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CsrCMatrix sub(CsrCMatrix B) {
        return null;
    }

    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CMatrix sub(double a) {
        return null;
    }

    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CMatrix sub(CNumber a) {
        return null;
    }

    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CsrCMatrix transpose() {
        return null;
    }

    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     *
     * @return The transpose of this tensor.
     */
    @Override
    public CsrCMatrix T() {
        return null;
    }

    /**
     * Gets the element in this tensor at the specified indices.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public CNumber get(int... indices) {
        return null;
    }

    /**
     * Creates a copy of this tensor.
     *
     * @return A copy of this tensor.
     */
    @Override
    public CsrCMatrix copy() {
        return null;
    }

    /**
     * Computes the element-wise multiplication between two tensors.
     *
     * @param B Tensor to element-wise multiply to this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CsrCMatrix elemMult(CsrCMatrix B) {
        return null;
    }

    /**
     * Computes the element-wise division between two tensors.
     *
     * @param B Tensor to element-wise divide with this tensor.
     * @return The result of the element-wise tensor multiplication.
     * @throws IllegalArgumentException If this tensor and {@code B} do not have the same shape.
     */
    @Override
    public CsrCMatrix elemDiv(CMatrix B) {
        return null;
    }

    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        return 0;
    }

    /**
     * Computes the p-norm of this tensor.
     *
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    @Override
    public double norm(double p) {
        return 0;
    }

    /**
     * A factory for creating a complex sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CsrCMatrix makeTensor(Shape shape, CNumber[] entries, int[][] indices) {
        return null;
    }

    /**
     * A factory for creating a real sparse tensor.
     *
     * @param shape   Shape of the sparse tensor to make.
     * @param entries Non-zero entries of the sparse tensor to make.
     * @param indices Non-zero indices of the sparse tensor to make.
     * @return A tensor created from the specified parameters.
     */
    @Override
    protected CsrMatrix makeRealTensor(Shape shape, double[] entries, int[][] indices) {
        return null;
    }

    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor which is equivalent to this sparse tensor.
     */
    @Override
    public CMatrix toDense() {
        return null;
    }


    /**
     * Converts this {@link CsrMatrix CSR matrix} to an equivalent {@link CooMatrix COO matrix}.
     * @return A {@link CooMatrix COO matrix} equivalent to this {@link CsrMatrix CSR matrix}.
     */
    public CooCMatrix toCoo() {

        CNumber[] dest = entries.clone();
        int[] destRowIdx = new int[entries.length];
        int[] destColIdx = colIndices.clone();

        for(int i=0; i<numRows; i++) {
            int stop = rowPointers[i+1];

            for(int j=rowPointers[i]; j<stop; j++) {
                destRowIdx[j] = i;
            }
        }

        return new CooCMatrix(shape.copy(), dest, destRowIdx, destColIdx);
    }


    /**
     * Formats this sparse matrix as a human-readable string.
     * @return A human-readable string representing this sparse matrix.
     */
    public String toString() {
        int size = nonZeroEntries;
        StringBuilder result = new StringBuilder(String.format("Full Shape: %s\n", shape));
        result.append("Non-zero entries: [");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        if(entries.length > 0) {
            // Get entries up until the stopping point.
            for(int i=0; i<stopIndex; i++) {
                value = StringUtils.ValueOfRound(entries[i], PrintOptions.getPrecision());
                width = PrintOptions.getPadding() + value.length();
                value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            if(stopIndex < size-1) {
                width = PrintOptions.getPadding() + 3;
                value = "...";
                value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            // Get last entry now
            value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");

        result.append("Row Pointers: ").append(Arrays.toString(rowPointers)).append("\n");
        result.append("Col Indices: ").append(Arrays.toString(colIndices));

        return result.toString();
    }
}
