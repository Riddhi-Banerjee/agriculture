def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #ffffff;
        }

        h1 {
            text-align: center;
            color: #ffffff;
            font-size: 42px;
            font-weight: 700;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
        }

        h2, h3, h4, h5, h6, p, label {
            color: #ffffff;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
        }

        .glass {
            background: rgba(255, 255, 255, 0.12);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
            color: #ffffff;
        }

        .stSlider > div > div > div > span {
            color: #ffeb3b !important; /* bright yellow for slider labels */
        }

        .stNumberInput>div>div>input {
            color: #ffffff;
            font-weight: bold;
        }

        .stSelectbox>div>div>div>div {
            color: #ffffff;
            font-weight: bold;
        }

        .stButton>button {
            background: linear-gradient(90deg, #00c853, #64dd17);
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            transition: 0.3s;
        }

        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #64dd17, #00c853);
        }

        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )
