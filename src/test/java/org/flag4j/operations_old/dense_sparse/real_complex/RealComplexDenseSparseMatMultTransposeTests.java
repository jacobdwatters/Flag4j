package org.flag4j.operations_old.dense_sparse.real_complex;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.sparse.CooCMatrix;
import org.flag4j.arrays_old.sparse.CooMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealComplexDenseSparseMatMultTransposeTests {
    static double[][] rdEntries = {};
    static CNumber[][] cdEntries = {};

    static double[] rspEntries = {};
    static CNumber[] cspEntries ={};

    static int[][] rindices;
    static int[][] cindices;

    static Shape rspShape;
    static Shape cspShape;

    static MatrixOld realDense;
    static CMatrixOld complexDense;
    static CooMatrix realSp;
    static CooCMatrix complexSp;

    static CMatrixOld exp;

    @BeforeAll
    static void setup() {
        rdEntries = new double[][]{
                {1.234, -00.024, 0, 1},
                {100.4, 5.14, -1.444, 0.041},
                {1.45, 985.1, -75.1, 4}
        };
        cdEntries = new CNumber[][]{
                {new CNumber(-0.24, 14.5), new CNumber(0.425)},
                {new CNumber(8.33, -84.4), CNumber.ZERO},
                {new CNumber(4.5, -9.24), new CNumber(0, -85.2)},
                {new CNumber("1.345"), new CNumber("-85.445+15.5i")}
        };

        rspShape = new Shape(2, 3).swapAxes(0, 1);
        cspShape = new Shape(4, 2).swapAxes(0, 1);

        rindices = new int[][]{{0, 2, 1}, {0, 0, 1}};
        cindices = new int[][]{{0, 1, 1}, {0, 2, 3}};

        rspEntries = new double[]{1.445, -9.25, 1.5};
        cspEntries = new CNumber[]{new CNumber(51.5, -0.42),
                new CNumber(-5.25, 15), new CNumber(3.45)};

        realDense = new MatrixOld(rdEntries);
        complexDense = new CMatrixOld(cdEntries);
        realSp = new CooMatrix(rspShape, rspEntries, rindices[0], rindices[1]);
        complexSp = new CooCMatrix(cspShape, cspEntries, cindices[0], cindices[1]);
    }


    @Test
    void realDenseComplexSpTestCase() {
        // ---------------------- sub-case 1 ----------------------
        exp = realDense.mult(new CooCMatrix(complexSp.shape.swapAxes(0, 1),
                complexSp.entries, complexSp.colIndices, complexSp.rowIndices));
        assertEquals(exp, realDense.multTranspose(complexSp));
    }


    @Test
    void complexDenseRealSpTestCase() {
        // ---------------------- sub-case 1 ----------------------
        exp = complexDense.mult(new CooMatrix(realSp.shape.swapAxes(0, 1),
                realSp.entries, realSp.colIndices, realSp.rowIndices));
        assertEquals(exp, complexDense.multTranspose(realSp));
    }
}
