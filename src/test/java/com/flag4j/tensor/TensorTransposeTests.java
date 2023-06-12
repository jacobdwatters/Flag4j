package com.flag4j.tensor;

import com.flag4j.Shape;
import com.flag4j.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorTransposeTests {

    static int[] aAxes;

    static double[] aEntries;
    static double[] expEntries;

    static Shape aShape;
    static Shape expShape;

    static Tensor A;
    static Tensor exp;

    @BeforeEach
    void setup() {
        aEntries = new double[]{
                1.4415, 235.61, -0.00024, 1.0, -85.1, 1.345,
                0.014, -140.0, 1.5, 51.0, 6.1, -0.00014};
        aShape = new Shape(3, 2, 1, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void transposeTestCase() {
        // -------------------- Sub-case 1 --------------------
        expEntries = new double[]{
                1.4415, -85.1, 1.5, -0.00024, 0.014, 6.1,
                235.61, 1.345, 51.0, 1.0, -140.0, -0.00014};
        expShape = new Shape(2, 2, 1, 3);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.transpose());

        // -------------------- Sub-case 2 --------------------
        aAxes = new int[]{0, 2, 3, 1};
        expEntries = new double[]{
                1.4415, -0.00024, 235.61, 1.0, -85.1, 0.014,
                1.345, -140.0, 1.5, 6.1, 51.0, -0.00014
        };
        expShape = new Shape(3, 1, 2, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.transpose(aAxes));

        // -------------------- Sub-case 3 --------------------
        aAxes = new int[]{3, 2, 1, 0};
        expEntries = new double[]{
                1.4415, -85.1, 1.5, -0.00024, 0.014, 6.1,
                235.61, 1.345, 51.0, 1.0, -140.0, -0.00014
        };
        expShape = new Shape(2, 1, 2, 3);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.transpose(aAxes));

        // -------------------- Sub-case 4 --------------------
        expEntries = new double[]{
                1.4415, -0.00024, 235.61, 1.0, -85.1,
                0.014, 1.345, -140.0, 1.5, 6.1, 51.0, -0.00014
        };
        expShape = new Shape(3, 2, 1, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.transpose(3, 1));

        // -------------------- Sub-case 5 --------------------
        aAxes = new int[]{3, 2, 1, 2};
        assertThrows(IllegalArgumentException.class, ()->A.transpose(aAxes));

        // -------------------- Sub-case 6 --------------------
        aAxes = new int[]{3, 2, 1};
        assertThrows(IllegalArgumentException.class, ()->A.transpose(aAxes));

        // -------------------- Sub-case 7 --------------------
        aAxes = new int[]{0, 1, 3, 2, 4};
        assertThrows(IllegalArgumentException.class, ()->A.transpose(aAxes));

        // -------------------- Sub-case 8 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.transpose(-1, 0));

        // -------------------- Sub-case 9 --------------------
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.transpose(1, 6));
    }
}
