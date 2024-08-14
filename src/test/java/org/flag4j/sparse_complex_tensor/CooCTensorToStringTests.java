package org.flag4j.sparse_complex_tensor;

import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.io.PrintOptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CooCTensorToStringTests {

    static CooCTensor A;
    static Shape aShape;
    static CNumber[] aEntries;
    static int[][] aIndices;

    String exp;

    @AfterEach
    void reset() {
        PrintOptions.resetAll();
    }

    @Test
    void cooCTensorToStringTests() {
        // ----------------------- Sub-case 1 -----------------------
        aShape = new Shape(5, 4, 2, 3, 15);
        aEntries = new CNumber[]{new CNumber(0.8103, 0.0203), new CNumber(0.5684, 0.4151), new CNumber(0.9044, 0.8734), new CNumber(0.201, 0.7032), new CNumber(0.9682, 0.2723), new CNumber(0.4699, 0.8203), new CNumber(0.3871, 0.3395), new CNumber(0.7851, 0.3768), new CNumber(0.2315, 0.7695), new CNumber(0.8333, 0.8837), new CNumber(0.0398, 0.559), new CNumber(0.0405, 0.9707), new CNumber(0.488, 0.8343), new CNumber(0.2441, 0.7806), new CNumber(0.3995, 0.6793), new CNumber(0.3689, 0.6126), new CNumber(0.0767, 0.9631), new CNumber(0.8007, 0.4023)};
        aIndices = new int[][]{
                {0, 0, 0, 0, 0},
                {0, 1, 0, 0, 7},
                {0, 2, 1, 2, 4},
                {0, 3, 0, 0, 11},
                {0, 3, 1, 0, 10},
                {0, 3, 1, 2, 6},
                {1, 1, 1, 0, 7},
                {2, 0, 0, 0, 8},
                {2, 0, 0, 0, 10},
                {2, 2, 0, 2, 12},
                {2, 2, 1, 1, 6},
                {3, 0, 0, 0, 2},
                {3, 0, 1, 1, 1},
                {4, 0, 0, 2, 10},
                {4, 0, 1, 0, 7},
                {4, 2, 1, 1, 8},
                {4, 3, 0, 0, 6},
                {4, 3, 1, 2, 13}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        exp = "Shape: (5, 4, 2, 3, 15)\n" +
                "Non-zero Entries: [ 0.8103 + 0.0203i  0.5684 + 0.4151i  0.9044 + 0.8734i  0.201 + 0.7032i  0.9682 + 0.2723i  0.4699 + 0.8203i  0.3871 + 0.3395i  0.7851 + 0.3768i  0.2315 + 0.7695i  ...  0.8007 + 0.4023i ]\n" +
                "Non-zero Indices: [ [ 0  0  0  0  0 ]  \n" +
                "                    [ 0  1  0  0  7 ]  \n" +
                "                    [ 0  2  1  2  4 ]  \n" +
                "                    [ 0  3  0  0  11 ]  \n" +
                "                    [ 0  3  1  0  10 ]  \n" +
                "                    [ 0  3  1  2  6 ]  \n" +
                "                    [ 1  1  1  0  7 ]  \n" +
                "                    [ 2  0  0  0  8 ]  \n" +
                "                    [ 2  0  0  0  10 ]  \n" +
                "                     ...  \n" +
                "                    [ 4  3  1  2  13 ]  ]";
        assertEquals(exp, A.toString());

        // ----------------------- Sub-case 2 -----------------------
        PrintOptions.setMaxRows(15);
        PrintOptions.setMaxColumns(3);
        PrintOptions.setPrecision(3);

        exp = "Shape: (5, 4, 2, 3, 15)\n" +
                "Non-zero Entries: [ 0.81 + 0.02i  0.568 + 0.415i  ...  0.801 + 0.402i ]\n" +
                "Non-zero Indices: [ [ 0  0  ...  0 ]  \n" +
                "                    [ 0  1  ...  7 ]  \n" +
                "                    [ 0  2  ...  4 ]  \n" +
                "                    [ 0  3  ...  11 ]  \n" +
                "                    [ 0  3  ...  10 ]  \n" +
                "                    [ 0  3  ...  6 ]  \n" +
                "                    [ 1  1  ...  7 ]  \n" +
                "                    [ 2  0  ...  8 ]  \n" +
                "                    [ 2  0  ...  10 ]  \n" +
                "                    [ 2  2  ...  12 ]  \n" +
                "                    [ 2  2  ...  6 ]  \n" +
                "                    [ 3  0  ...  2 ]  \n" +
                "                    [ 3  0  ...  1 ]  \n" +
                "                    [ 4  0  ...  10 ]  \n" +
                "                     ...  \n" +
                "                    [ 4  3  ...  13 ]  ]";
        assertEquals(exp, A.toString());
    }
}
