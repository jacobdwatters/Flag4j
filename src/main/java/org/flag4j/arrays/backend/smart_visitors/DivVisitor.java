/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.arrays.backend.smart_visitors;

import org.flag4j.algebraic_structures.Field;
import org.flag4j.algebraic_structures.Ring;
import org.flag4j.algebraic_structures.Semiring;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.*;
import org.flag4j.arrays.sparse.*;

public class DivVisitor extends MatrixVisitor<MatrixMixin<?, ?, ?, ?>> {

    public DivVisitor(MatrixMixin<?, ?, ?, ?> other) {
        super(other);
    }

    @Override
    public MatrixMixin<?, ?, ?, ?> visit(Matrix matrix) {
        if(other instanceof Matrix)
            return matrix.div((Matrix) other);
        else if(other instanceof CMatrix)
            return matrix.div((CMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CMatrix matrix) {
        if(other instanceof Matrix)
            return matrix.div((Matrix) other);
        else if(other instanceof CMatrix)
            return matrix.div((CMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CooMatrix matrix) {
        if(other instanceof CooMatrix)
            return matrix.div((CooMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CooCMatrix matrix) {
        if(other instanceof CooCMatrix)
            return matrix.div((CooCMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CsrMatrix matrix) {
        if(other instanceof CsrMatrix)
            return matrix.div((CsrMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CsrCMatrix matrix) {
        if(other instanceof CsrCMatrix)
            return matrix.div((CsrCMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }

    @Override
    public <U extends Field<U>> MatrixMixin<?, ?, ?, ?> visit(FieldMatrix<U> matrix) {
        if(other instanceof FieldMatrix<?>)
            return matrix.div((FieldMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends Field<U>> MatrixMixin<?, ?, ?, ?> visit(CooFieldMatrix<U> matrix) {
        if(other instanceof CooFieldMatrix<?>)
            return matrix.div((CooFieldMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends Field<U>> MatrixMixin<?, ?, ?, ?> visit(CsrFieldMatrix<U> matrix) {
        if(other instanceof CsrFieldMatrix<?>)
            return matrix.div((CsrFieldMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends Semiring<U>> MatrixMixin<?, ?, ?, ?> visit(SemiringMatrix<U> matrix) {
        if(other instanceof SemiringMatrix<?>)
            return matrix.div((SemiringMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends Semiring<U>> MatrixMixin<?, ?, ?, ?> visit(CooSemiringMatrix<U> matrix) {
        if(other instanceof CooSemiringMatrix<?>)
            return matrix.div((CooSemiringMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends Semiring<U>> MatrixMixin<?, ?, ?, ?> visit(CsrSemiringMatrix<U> matrix) {
        if(other instanceof CsrSemiringMatrix<?>)
            return matrix.div((CsrSemiringMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends Ring<U>> MatrixMixin<?, ?, ?, ?> visit(RingMatrix<U> matrix) {
        if(other instanceof RingMatrix<?>)
            return matrix.div((RingMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends Ring<U>> MatrixMixin<?, ?, ?, ?> visit(CooRingMatrix<U> matrix) {
        if(other instanceof CooRingMatrix<?>)
            return matrix.div((CooRingMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends Ring<U>> MatrixMixin<?, ?, ?, ?> visit(CsrRingMatrix<U> matrix) {
        if(other instanceof CsrRingMatrix<?>)
            return matrix.div((CsrRingMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }
}
