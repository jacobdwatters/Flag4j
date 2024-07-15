package org.flag4j.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.linalg.ops.DirectSum;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixDirectSumTests {

    Shape sparseShape;
    int[] rowIndices, colIndices;

    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realDirectSumTestCase() {
        double[][] bEntries;
        Matrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {1, -3.23},
                {324, 5.234},
                {-74.13, 44.5}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{{new CNumber("9.234-0.864i"), new CNumber("58.1+3.0i"), new CNumber("-984.0-72.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.0"), new CNumber("0.0"), new CNumber("0.0+87.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("1.0"), new CNumber("-3.23")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("324.0"), new CNumber("5.234")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("-74.13"), new CNumber("44.5")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void realSparseDirectSumTestCase() {
        double[] bEntries;
        CooMatrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new double[]{2.456, -7.41};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{1, 0};
        sparseShape = new Shape(3, 2);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{{new CNumber("9.234-0.864i"), new CNumber("58.1+3.0i"), new CNumber("-984.0-72.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.0"), new CNumber("0.0"), new CNumber("0.0+87.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("2.456")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("-7.41"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void complexDirectSumTestCase() {
        CNumber[][] bEntries;
        CMatrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.0+9.435i"), new CNumber("-3.23-8.234i")},
                {new CNumber("0.0+324.0i"), new CNumber("5.234-8.4i")},
                {new CNumber("-74.13+475.145i"), new CNumber("44.5+8.345i")}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{{new CNumber("9.234-0.864i"), new CNumber("58.1+3.0i"), new CNumber("-984.0-72.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.0"), new CNumber("0.0"), new CNumber("0.0+87.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("1.0+9.435i"), new CNumber("-3.23-8.234i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0+324.0i"), new CNumber("5.234-8.4i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("-74.13+475.145i"), new CNumber("44.5+8.345i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void complexSparseDirectSumTestCase() {
        CNumber[] bEntries;
        CooCMatrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6567, -6344.256), new CNumber(Double.NEGATIVE_INFINITY, 234.56)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{1, 0};
        sparseShape = new Shape(3, 2);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{{new CNumber("9.234-0.864i"), new CNumber("58.1+3.0i"), new CNumber("-984.0-72.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.0"), new CNumber("0.0"), new CNumber("0.0+87.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber(234.6567, -6344.256)},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber(Double.NEGATIVE_INFINITY, 234.56), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.directSum(A, B));
    }


    @Test
    void realInvDirectSumTestCase() {
        double[][] bEntries;
        Matrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new double[][]{
                {1, -3.23},
                {324, 5.234},
                {-74.13, 44.5}};
        B = new Matrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("1.0"), new CNumber("-3.23")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("324.0"), new CNumber("5.234")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("-74.13"), new CNumber("44.5")},
                {new CNumber("9.234-0.864i"), new CNumber("58.1+3.0i"), new CNumber("-984.0-72.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.0"), new CNumber("0.0"), new CNumber("0.0+87.3i"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }


    @Test
    void realInvSparseDirectSumTestCase() {
        double[] bEntries;
        CooMatrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new double[]{2.456, -7.41};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{1, 0};
        sparseShape = new Shape(3, 2);
        B = new CooMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("2.456")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("-7.41"), new CNumber("0.0")},
                {new CNumber("9.234-0.864i"), new CNumber("58.1+3.0i"), new CNumber("-984.0-72.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.0"), new CNumber("0.0"), new CNumber("0.0+87.3i"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }


    @Test
    void complexInvDirectSumTestCase() {
        CNumber[][] bEntries;
        CMatrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new CNumber[][]{
                {new CNumber("1.0+9.435i"), new CNumber("-3.23-8.234i")},
                {new CNumber("0.0+324.0i"), new CNumber("5.234-8.4i")},
                {new CNumber("-74.13+475.145i"), new CNumber("44.5+8.345i")}};
        B = new CMatrix(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("1.0+9.435i"), new CNumber("-3.23-8.234i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0+324.0i"), new CNumber("5.234-8.4i")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("-74.13+475.145i"), new CNumber("44.5+8.345i")},
                {new CNumber("9.234-0.864i"), new CNumber("58.1+3.0i"), new CNumber("-984.0-72.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.0"), new CNumber("0.0"), new CNumber("0.0+87.3i"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }


    @Test
    void complexInvSparseDirectSumTestCase() {
        CNumber[] bEntries;
        CooCMatrix B;

        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(9.234, -0.864), new CNumber(58.1, 3), new CNumber(-984, -72.3)},
                {new CNumber(1), new CNumber(), new CNumber(0, 87.3)}
        };
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6567, -6344.256), new CNumber(Double.NEGATIVE_INFINITY, 234.56)};
        rowIndices = new int[]{0, 2};
        colIndices = new int[]{1, 0};
        sparseShape = new Shape(3, 2);
        B = new CooCMatrix(sparseShape, bEntries, rowIndices, colIndices);
        expEntries = new CNumber[][]{
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber(234.6567, -6344.256)},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("0.0"), new CNumber("0.0"), new CNumber("0.0"), new CNumber(Double.NEGATIVE_INFINITY, 234.56), new CNumber("0.0")},
                {new CNumber("9.234-0.864i"), new CNumber("58.1+3.0i"), new CNumber("-984.0-72.3i"), new CNumber("0.0"), new CNumber("0.0")},
                {new CNumber("1.0"), new CNumber("0.0"), new CNumber("0.0+87.3i"), new CNumber("0.0"), new CNumber("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, DirectSum.invDirectSum(A, B));
    }
}
