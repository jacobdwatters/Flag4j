package org.flag4j.linalg.transformations;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GivensTests {

    double[] vEntriesReal;
    CNumber[] vEntriesComplex;

    VectorOld vReal;
    CVectorOld vComplex;

    double[][] expEntriesReal;
    MatrixOld expReal;

    CNumber[][] expEntriesComplex;
    CMatrixOld expComplex;


    @Test
    void generalRotatorTestCase() {
        int size, i, j;
        double theta;

        // ------------------- Sub-case 1 -------------------
        size = 105;
        j = 4;
        i = 91;
        theta = Math.PI/4.0;
        expReal = MatrixOld.I(size);
        expReal.set(Math.cos(theta), i, i);
        expReal.set(Math.cos(theta), j, j);
        expReal.set(Math.sin(theta), i, j);
        expReal.set(-Math.sin(theta), j, i);

        assertEquals(expReal, GivensOld.getGeneralRotator(size, i, j, theta));

        // ------------------- Sub-case 2 -------------------
        size = 3;
        j = 1;
        i = 2;
        theta = 0.0;
        expReal = MatrixOld.I(size);
        expReal.set(Math.cos(theta), i, i);
        expReal.set(Math.cos(theta), j, j);
        expReal.set(Math.sin(theta), i, j);
        expReal.set(-Math.sin(theta), j, i);

        assertEquals(expReal, GivensOld.getGeneralRotator(size, i, j, theta));

        // ------------------- Sub-case 3 -------------------
        assertThrows(IllegalArgumentException.class, ()-> GivensOld.getGeneralRotator(5, 3, 3, 2));
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getGeneralRotator(5, 1, 6, 2));
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getGeneralRotator(5, 6, 0, 2));
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getGeneralRotator(5, -1, 2, 2));
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getGeneralRotator(5, 1, -5, 2));
    }


    @Test
    void realRotatorTestCase() {
        int i;

        // ------------------- Sub-case 1 -------------------
        i = 4;
        vEntriesReal = new double[]{1.56, 1.3567, -0.02456, 103.6, -992.255, 88.156};
        vReal = new VectorOld(vEntriesReal);

        expEntriesReal = new double[][]{
                {0.001572174564045712, 0.0, 0.0, 0.0, -0.9999987641328064, 0.0},
                {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                {0.9999987641328064, 0.0, 0.0, 0.0, 0.001572174564045712, 0.0},
                {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
        };
        expReal = new MatrixOld(expEntriesReal);

        assertEquals(expReal, GivensOld.getRotator(vReal, i));

        // ------------------- Sub-case 2 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getRotator(vReal, -1));
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getRotator(vReal, vReal.size));
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getRotator(vReal, vReal.size + 3));
    }


    @Test
    void complexRotatorTestCase() {
        int i;

        // ------------------- Sub-case 1 -------------------
        i = 1;
        vEntriesComplex = new CNumber[]{new CNumber(1.456), new CNumber(-2, 15.6),
                new CNumber(2.6, -0.2), CNumber.ZERO};
        vComplex = new CVectorOld(vEntriesComplex);

        expEntriesComplex = new CNumber[][]{
                {new CNumber(0.09095028651755654, 0.0), new CNumber(-0.12493171224939086, 0.9744673555452487), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.12493171224939086, -0.9744673555452487), new CNumber(0.09095028651755654, -0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(1.0, 0.0), new CNumber(0.0, 0.0)},
                {new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(0.0, 0.0), new CNumber(1.0, 0.0)}
        };
        expComplex = new CMatrixOld(expEntriesComplex);

        assertEquals(expComplex, GivensOld.getRotator(vComplex, i));

        // ------------------- Sub-case 2 -------------------
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getRotator(vComplex, -1));
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getRotator(vComplex, vComplex.size));
        assertThrows(IndexOutOfBoundsException.class, ()-> GivensOld.getRotator(vComplex, vComplex.size + 3));
    }


    @Test
    void real2x2RotatorTestCase() {
        // ------------------- Sub-case 1 -------------------
        vEntriesReal = new double[]{1.56, 1.3567};
        vReal = new VectorOld(vEntriesReal);

        expEntriesReal = new double[][]{
                {0.7545628263597614, 0.6562278118732616},
                {-0.6562278118732616, 0.7545628263597614}
        };
        expReal = new MatrixOld(expEntriesReal);

        assertEquals(expReal, GivensOld.get2x2Rotator(vReal));

        // ------------------- Sub-case 2 -------------------
        vEntriesReal = new double[]{1.56, 0};
        vReal = new VectorOld(vEntriesReal);

        expEntriesReal = new double[][]{
                {1.0, 0.0},
                {-0.0, 1.0}
        };
        expReal = new MatrixOld(expEntriesReal);
        
        assertEquals(expReal, GivensOld.get2x2Rotator(vReal));

        // ------------------- Sub-case 3 -------------------
        vEntriesReal = new double[]{0, -9.3};
        vReal = new VectorOld(vEntriesReal);

        expEntriesReal = new double[][]{
                {0.0, -1.0},
                {1.0, 0.0}
        };
        expReal = new MatrixOld(expEntriesReal);

        assertEquals(expReal, GivensOld.get2x2Rotator(vReal));

        // ------------------- Sub-case 4 -------------------
        vEntriesReal = new double[]{0, 0};
        vReal = new VectorOld(vEntriesReal);

        expEntriesReal = new double[][]{
                {1.0, 0.0},
                {-0.0, 1.0}
        };
        expReal = new MatrixOld(expEntriesReal);

        assertEquals(expReal, GivensOld.get2x2Rotator(vReal));

        // ------------------- Sub-case 5 -------------------
        assertThrows(IllegalArgumentException.class, ()-> GivensOld.get2x2Rotator(new VectorOld(1)));
        assertThrows(IllegalArgumentException.class, ()-> GivensOld.get2x2Rotator(new VectorOld(5)));
    }


    @Test
    void complex2x2RotatorTestCase() {
        // ------------------- Sub-case 1 -------------------
        vEntriesComplex = new CNumber[]{new CNumber(2.56, -9.53), new CNumber(3.6, 0.00134)};
        vComplex = new CVectorOld(vEntriesComplex);

        expEntriesComplex = new CNumber[][]{
                {new CNumber(0.24371614282534812, 0.9072714223146748), new CNumber(0.3427258258481458, 1.2757016851014318E-4)},
                {new CNumber(-0.3427258258481458, -1.2757016851014318E-4), new CNumber(0.24371614282534812, -0.9072714223146748)}
        };
        expComplex = new CMatrixOld(expEntriesComplex);

        assertEquals(expComplex, GivensOld.get2x2Rotator(vComplex));

        // ------------------- Sub-case 2 -------------------
        assertThrows(IllegalArgumentException.class, ()-> GivensOld.get2x2Rotator(new CVectorOld(1)));
        assertThrows(IllegalArgumentException.class, ()-> GivensOld.get2x2Rotator(new CVectorOld(5)));
    }
}
