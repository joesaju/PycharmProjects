{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08920cfc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fe257b5c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      video_id                                              title  \\\n",
      "0  kZfz5UlsHlQ                        DARLING, I (Official Video)   \n",
      "1  9txkGBj_trg  Call of Duty: Black Ops 7 | Gameplay Reveal Tr...   \n",
      "2  HVC_dBNUZGc                   gamescom Opening Night Live 2025   \n",
      "3  5dA094oAy-g     Twenty One Pilots - Drum Show (Official Video)   \n",
      "4  NZY5WiqeyQQ  Fallout Season 2 - Official Teaser Trailer | g...   \n",
      "5  QtkHiB7WKf4                  Kirby Air Riders Direct 8.19.2025   \n",
      "6  K5Sz9uw6me8  Nino Paid - 3 Peat (ft. 1900Rugrat and VonOff1...   \n",
      "7  IRjuAemfaQE  Sekiro: No Defeat | Official Trailer | Crunchy...   \n",
      "8  e_ppaeEbRKo        steal a brainrot admin abuse (taco tuesday)   \n",
      "9  HNVmORZhgxI               T-Pain - Bartender (Lyrics) ft. Akon   \n",
      "\n",
      "                                         description          published_at  \\\n",
      "0  directed by TYLER OKONMA\\ndp: LUIS \"PANCH\" PER...  2025-08-19T17:09:11Z   \n",
      "1  Call of Duty®: Black Ops 7 redefines the franc...  2025-08-19T18:08:38Z   \n",
      "2  Live from Cologne – the big opening show of ga...  2025-08-19T20:15:33Z   \n",
      "3  Official video for the new single “Drum Show” ...  2025-08-18T17:30:06Z   \n",
      "4  Watch the Season 2 Teaser Trailer for Fallout,...  2025-08-19T18:55:58Z   \n",
      "5  Join us on Aug 19 at 6 a.m. PT for a Kirby Air...  2025-08-19T13:58:38Z   \n",
      "6  The Official Youtube Channel For Recording Art...  2025-08-19T19:00:27Z   \n",
      "7  Sekiro: No Defeat is coming exclusively to Cru...  2025-08-19T18:19:17Z   \n",
      "8  roblox steal a brainrot admin abuse live (taco...  2025-08-19T22:53:04Z   \n",
      "9  T-Pain - Bartender (Lyrics) ft. Akon\\n🎧 Listen...  2025-08-19T00:00:17Z   \n",
      "\n",
      "                 channel_id        channel_title  \\\n",
      "0  UCsQBsZJltmLzlsJNG7HevBg   Tyler, The Creator   \n",
      "1  UC9YydG57epLqxA9cTzZXSeQ         Call of Duty   \n",
      "2  UCHo_GVNoKNqfJx6zUGRd6YQ             gamescom   \n",
      "3  UCBQZwaNPFfJ1gZ1fLZpAEGw    twenty one pilots   \n",
      "4  UCKy1dAqELo0zrOtPkf0eTMw                  IGN   \n",
      "5  UCGIY_O-8vW4rfX98KlMkvRg  Nintendo of America   \n",
      "6  UC2hFKZgVRe1b2CpULVgaCDA            Nino Paid   \n",
      "7  UC6pGDc4bFGD1_36IKv3FnYg          Crunchyroll   \n",
      "8  UCxsk7hqE_CwZWGEJEkGanbA           KreekCraft   \n",
      "9  UCvWhPqiJBsdsZmM9cEzQorg           SoundKream   \n",
      "\n",
      "                                                tags category_id  view_count  \\\n",
      "0  chromakopia,tyler the creator,darling i,teezo ...           2   1042576.0   \n",
      "1                        call of duty,cod,activision          20   5991411.0   \n",
      "2                                                NaN          24   1540057.0   \n",
      "3  twenty one pilots,twenty one pilots official,n...          10   1973975.0   \n",
      "4  ign,gamescom,gamescom 2025,game trailer,game t...          20    831185.0   \n",
      "5  nintendo,game,gameplay,fun,video game,action,a...          20   1294325.0   \n",
      "6                                                NaN          10     77798.0   \n",
      "7  sekiro,sekiro anime,sekiro shadows die twice,s...           1    947397.0   \n",
      "8                                                NaN          20   1183647.0   \n",
      "9  t-pain ft. akon - bartender (lyrics),t-pain - ...          10     46089.0   \n",
      "\n",
      "   like_count  favorite_count  comment_count   duration definition caption  \\\n",
      "0    181841.0             0.0        13958.0    PT4M49S         hd   FALSE   \n",
      "1     25012.0             0.0        13127.0    PT2M20S         hd   FALSE   \n",
      "2     35942.0             0.0          667.0    PT3H14S         hd   FALSE   \n",
      "3    212769.0             0.0        14911.0    PT3M36S         hd   FALSE   \n",
      "4     58015.0             0.0         4846.0    PT2M43S         hd   FALSE   \n",
      "5     67730.0             0.0         5821.0   PT47M29S         hd   FALSE   \n",
      "6      8302.0             0.0          460.0    PT2M18S         hd   FALSE   \n",
      "7     95998.0             0.0         5654.0    PT1M13S         hd   FALSE   \n",
      "8     18579.0             0.0          586.0  PT2H25M8S         hd   FALSE   \n",
      "9       156.0             0.0            0.0     PT4M2S         hd   FALSE   \n",
      "\n",
      "  Unnamed: 15  \n",
      "0         NaN  \n",
      "1         NaN  \n",
      "2         NaN  \n",
      "3         NaN  \n",
      "4         NaN  \n",
      "5         NaN  \n",
      "6         NaN  \n",
      "7         NaN  \n",
      "8         NaN  \n",
      "9         NaN  \n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "new_data = pd.read_csv(r\"D:\\python\\New folder\\Linear regression\\Youtube Data.csv\")\n",
    "print(new_data.head(10))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
