package org.flag4j.operations_old.sparse.coo.complex;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.flag4j.operations_old.sparse.coo.complex.ComplexSparseMatrixMultiplication.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ComplexSparseMatMultTests {

    CNumber[][] expEntries;
    CNumber[] aEntries, bEntries, bVectorEntries;

    int[] aRowIndices, aColIndices, bRowIndices, bColIndices, indices;

    Shape aShape, bShape;

    CooCMatrix A, B;
    CooCVector bVector;
    CMatrixOld exp;

    @Test
    void matMultTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[]{new CNumber("1+3.45i"), new CNumber("9.43-8j")};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{2, 1};
        aShape = new Shape(4, 3);
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("1.34+13.4i"), new CNumber("9234")};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new CNumber[][]{{new CNumber("0.0"), new CNumber("9234.0+31857.300000000003i")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrixOld(expEntries);

        assertArrayEquals(exp.entries, standard(A.entries, A.rowIndices, A.colIndices, A.shape,
                B.entries, B.rowIndices, B.colIndices, B.shape));
        assertArrayEquals(exp.entries, concurrentStandard(A.entries, A.rowIndices, A.colIndices, A.shape,
                B.entries, B.rowIndices, B.colIndices, B.shape));
    }


    @Test
    void matVecMultTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        aEntries = new CNumber[]{new CNumber("1+3.45i"), new CNumber("7.9-105j"), new CNumber("9.43-8j")};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{2, 0, 1};
        aShape = new Shape(4, 3);
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new CNumber[]{new CNumber("1.34+13.4i")};
        indices = new int[]{1};
        bVector = new CooCVector(3, bEntries, indices);

        expEntries = new CNumber[][]{{new CNumber("0.0")},
                {new CNumber("0.0")},
                {new CNumber("119.8362+115.642i")},
                {new CNumber("0.0")}};
        exp = new CMatrixOld(expEntries);

        CNumber[] act = standardVector(A.entries, A.rowIndices, A.colIndices, A.shape,
                bVector.entries, bVector.indices);

        assertArrayEquals(exp.entries, standardVector(A.entries, A.rowIndices, A.colIndices, A.shape,
                bVector.entries, bVector.indices));
        assertArrayEquals(exp.entries, concurrentStandardVector(A.entries, A.rowIndices, A.colIndices, A.shape,
                bVector.entries, bVector.indices));
    }
}
