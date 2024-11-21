package org.flag4j.linalg.operations.sparse.coo.complex;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.junit.jupiter.api.Test;

import static org.flag4j.linalg.operations.sparse.coo.field_ops.CooFieldMatMult.*;

class ComplexSparseMatMultTests {

    Complex128[][] expEntries;
    Complex128[] aEntries, bEntries, bVectorEntries;

    int[] aRowIndices, aColIndices, bRowIndices, bColIndices, indices;

    Shape aShape, bShape;

    CooCMatrix A, B;
    CooCVector bVector;
    CMatrix exp;

    @Test
    void matMultTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        aEntries = new Complex128[]{new Complex128("1+3.45i"), new Complex128("9.43-8j")};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{2, 1};
        aShape = new Shape(4, 3);
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[]{new Complex128("1.34+13.4i"), new Complex128("9234")};
        bRowIndices = new int[]{0, 2};
        bColIndices = new int[]{0, 1};
        bShape = new Shape(3, 2);
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new Complex128[][]{{new Complex128("0.0"), new Complex128("9234.0+31857.300000000003i")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertArrayEquals(exp.entries, standard(A.entries, A.rowIndices, A.colIndices, A.shape,
                B.entries, B.rowIndices, B.colIndices, B.shape));
        assertArrayEquals(exp.entries, concurrentStandard(A.entries, A.rowIndices, A.colIndices, A.shape,
                B.entries, B.rowIndices, B.colIndices, B.shape));
    }


    @Test
    void matVecMultTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        aEntries = new Complex128[]{new Complex128("1+3.45i"), new Complex128("7.9-105j"), new Complex128("9.43-8j")};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{2, 0, 1};
        aShape = new Shape(4, 3);
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[]{new Complex128("1.34+13.4i")};
        indices = new int[]{1};
        bVector = new CooCVector(3, bEntries, indices);

        expEntries = new Complex128[][]{{new Complex128("0.0")},
                {new Complex128("0.0")},
                {new Complex128("119.8362+115.642i")},
                {new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        Field<Complex128>[] act = standardVector(A.entries, A.rowIndices, A.colIndices, A.shape,
                bVector.entries, bVector.indices);

        assertArrayEquals(exp.entries, standardVector(A.entries, A.rowIndices, A.colIndices, A.shape,
                bVector.entries, bVector.indices));
        assertArrayEquals(exp.entries, concurrentStandardVector(A.entries, A.rowIndices, A.colIndices, A.shape,
                bVector.entries, bVector.indices));
    }
}
